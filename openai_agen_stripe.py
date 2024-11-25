import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import stripe
from typing import Optional
from stripe_agent_toolkit.langchain.toolkit import StripeAgentToolkit
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from mangum import Mangum

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize Stripe Agent Toolkit
stripe_agent_toolkit = StripeAgentToolkit(
    secret_key=os.getenv("STRIPE_SECRET_KEY"),
    configuration={
        "actions": {
            "subscriptions": {"list": True, "retrieve": True, "cancel": True},
            "refunds": {"create": True},
            "invoices": {"list": True, "retrieve": True},
        }
    }
)

# Define user input schema
class UserMessage(BaseModel):
    message: str
    email: str

# Define tools for LangChain agent
def list_subscriptions_tool(customer_id: str) -> str:
    try:
        subscriptions = stripe.Subscription.list(customer=customer_id)
        if not subscriptions["data"]:
            return "No active subscriptions found."
        return "\n".join(
            [f"Subscription ID: {sub['id']}, Plan: {sub['plan']['nickname']}" for sub in subscriptions["data"]]
        )
    except Exception as e:
        return f"Error listing subscriptions: {str(e)}"

def cancel_subscription_tool(customer_id: str,subscription_id: str) -> str:
    try:
        subscriptions = stripe.Subscription.list(customer=customer_id)
        if not subscriptions["data"]:
            return "No active subscriptions to cancel."
        stripe.Subscription.delete(subscription_id)
        return f"Subscription '{subscription_id}' has been successfully cancelled."
    except Exception as e:
        return f"Error cancelling subscription: {str(e)}"

def calculate_refund_tool(customer_id: str) -> str:
    try:
        subscriptions = stripe.Subscription.list(customer=customer_id)
        if not subscriptions["data"]:
            return "No active subscriptions to calculate refunds for."

        subscription = subscriptions["data"][0]
        amount_due = subscription["plan"]["amount"]
        current_period_end = subscription["current_period_end"]
        current_period_start = subscription["current_period_start"]

        import time
        total_days = (current_period_end - current_period_start) / 86400
        remaining_days = (current_period_end - time.time()) / 86400
        prorated_refund = (remaining_days / total_days) * amount_due

        return f"The estimated refund for the remaining period is {prorated_refund:.2f} units."
    except Exception as e:
        return f"Error calculating refund: {str(e)}"

def process_refund_tool(customer_id: str) -> str:
    try:
        invoices = stripe.Invoice.list(customer=customer_id)
        if not invoices["data"]:
            return "No invoices found to process a refund."
        last_invoice = invoices["data"][0]
        payment_intent_id = last_invoice["payment_intent"]

        refund = stripe.Refund.create(payment_intent=payment_intent_id)
        return f"Refund of {refund['amount'] / 100:.2f} {refund['currency']} has been successfully processed."
    except Exception as e:
        return f"Error processing refund: {str(e)}"

def get_customer_details_tool(customer_id: str) -> str:
    try:
        customer = stripe.Customer.retrieve(customer_id)
        return f"Customer Details: Name: {customer.get('name')}, Email: {customer.get('email')}, Balance: {customer.get('balance')}"
    except Exception as e:
        return f"Error retrieving customer details: {str(e)}"
    
def get_customer_id_by_email(email: str) -> Optional[str]:
    try:
        customers = stripe.Customer.list(email=email)
        if not customers["data"]:
            return None
        return customers["data"][0]["id"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving customer ID by email: {str(e)}")

# Initialize OpenAI LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    max_tokens=500,
)

# Define system prompt
system_prompt = """
You are a helpful customer support assistant for Stripe. You assist customers with managing their subscriptions, including listing subscriptions, cancelling subscriptions, calculating refunds, retrieving customer details, and processing refunds.

When interacting with the customer, always be polite and provide clear and concise answers.

Follow this workflow:
1. Greet the user.
2. Offer assistance by summarizing available options.
3. When the user requests to view active subscriptions, list them.
4. When the user wants to cancel a subscription, ask which one they want to cancel.
5. After the user specifies, ask for confirmation before proceeding with the cancellation.
6. If the user confirms, proceed to cancel the subscription and inform them of the successful cancellation.
7. After performing an action, ask if the user needs further assistance.

**Important:** **Do not perform any action without explicit user confirmation.** Ensure that every sensitive action, especially cancellations, is preceded by a confirmation step.

Maintain context across the conversation to handle sequential requests effectively.

**Example Conversation:**

**User:** Hey  
**Chatbot:** Hello! How can I assist you today? You can view your active subscriptions, cancel a subscription, calculate refunds, or retrieve your customer details.

**User:** I want to see my active subscriptions  
**Chatbot:** Sure, here are your active subscriptions:
- Subscription ID: sub_1QMZU0SCIn8smr1rrm15HV6F, Plan: Premium Plan (ending on 2024-12-31)
- Subscription ID: sub_2ABCXYZ12345, Plan: Basic Plan (ending on 2024-11-30). Would you like to cancel any of these?

**User:** I want to cancel my Premium Plan subscription  
**Chatbot:** You've selected to cancel Subscription ID: sub_1QMZU0SCIn8smr1rrm15HV6F, Premium Plan. **Are you sure you want to proceed with the cancellation? (yes/no)**

**User:** Yes, please  
**Chatbot:** Your subscription 'sub_1QMZU0SCIn8smr1rrm15HV6F' has been successfully cancelled. Can I help you with anything else?
"""


system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])


# Initialize a dictionary to store memory instances per user
user_memory_store = {}

# FastAPI endpoint
@app.post("/chat")
async def chat_with_agent(user_message: UserMessage):
    """
    Endpoint to handle user messages and process them with the agent.
    """
    try:
        email = user_message.email
        customer_id = get_customer_id_by_email(email)
        if not customer_id:
            raise HTTPException(status_code=404, detail="No customer found with the provided email.")
        
        # Retrieve or create memory for the user
        if email not in user_memory_store:
            user_memory_store[email] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        memory = user_memory_store[email]

        tools = [
            Tool(
                name="List Subscriptions",
                func=lambda _: list_subscriptions_tool(customer_id),
                description="Fetches and returns the list of active subscriptions for the customer."
            ),
            Tool(
                name="Cancel Subscription",
                func=lambda subscription_id: cancel_subscription_tool(customer_id, subscription_id),
                description="Cancels the specified subscription for the customer. **This tool should only be used after the user has confirmed the cancellation.** Provide the Subscription ID to proceed."
            ),
            Tool(
                name="Calculate Refund",
                func=lambda _: calculate_refund_tool(customer_id),
                description="Calculates and returns the estimated refund for the customer."
            ),
            Tool(
                name="Process Refund",
                func=lambda _: process_refund_tool(customer_id),
                description="Processes a refund for the customer's last payment."
            ),
            Tool(
                name="Customer Details",
                func=lambda _: get_customer_details_tool(customer_id),
                description="Fetches and returns the details of the customer."
            )
        ]


        agent_chain = initialize_agent(
            tools=tools,
            llm=llm,
            agent="chat-conversational-react-description",
            verbose=True,
            memory=memory,
            handle_parsing_errors=True,
            prompt=chat_prompt
        )

        # Run the agent with a fallback mechanism for incomplete output
        try:
            response = agent_chain.run(user_message.message)
        except Exception as parsing_error:
            response = "I'm sorry, there was an issue processing your request. Please try again."

        print(f"Agent Response for {email}: {response}")
        return {"response": response}
    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# Run the app
handler = Mangum(app)

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
