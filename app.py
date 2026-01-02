import os
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import stripe

app = FastAPI()

stripe.api_key = os.environ["STRIPE_SECRET_KEY"]
WEBHOOK_SECRET = os.environ["STRIPE_WEBHOOK_SECRET"]

# Simple in-memory store (OK for testing). For real production, swap to a DB.
PRO_EMAILS = set()

@app.get("/")
def home():
    return {"ok": True}

@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload, sig_header=sig_header, secret=WEBHOOK_SECRET
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    # When subscription is created/active
    if event["type"] in ("checkout.session.completed",):
        session = event["data"]["object"]
        email = session.get("customer_details", {}).get("email")
        if email:
            PRO_EMAILS.add(email.lower())

    # If you want to handle cancellations later, add customer.subscription.deleted, etc.

    return JSONResponse({"received": True})

@app.get("/pro/check")
def check_pro(email: str):
    if not email:
        return {"pro": False}
    return {"pro": email.lower() in PRO_EMAILS}
