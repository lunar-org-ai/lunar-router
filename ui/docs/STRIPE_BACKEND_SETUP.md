# Stripe Backend Implementation Guide

## Overview
This guide shows how to implement the backend endpoints required for Stripe payment processing using your live API key.

## Required Endpoints

### 1. Create Payment Intent
**Endpoint:** `POST /api/create-payment-intent`

```javascript
const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);

app.post('/api/create-payment-intent', async (req, res) => {
  try {
    const { amount, currency, metadata } = req.body;
    
    // Validate amount (minimum $1.00)
    if (amount < 100) {
      return res.status(400).json({ error: 'Minimum amount is $1.00' });
    }
    
    // Create payment intent with Stripe
    const paymentIntent = await stripe.paymentIntents.create({
      amount,
      currency,
      metadata,
      automatic_payment_methods: {
        enabled: true,
      },
    });

    res.json({
      client_secret: paymentIntent.client_secret,
      id: paymentIntent.id,
    });
  } catch (error) {
    console.error('Error creating payment intent:', error);
    res.status(400).json({ error: error.message });
  }
});
```

### 2. Confirm Payment (Optional)
**Endpoint:** `POST /api/confirm-payment`

```javascript
app.post('/api/confirm-payment', async (req, res) => {
  try {
    const { paymentIntentId, amount } = req.body;
    
    // Here you would:
    // 1. Verify the payment was successful with Stripe
    // 2. Add credits to the user's account in your database
    // 3. Log the transaction
    
    const credits = Math.round(amount / 100); // $1 = 1 credit
    
    // Example: Update user credits in database
    // await updateUserCredits(userId, credits);
    
    res.json({ 
      success: true,
      credits_added: credits 
    });
  } catch (error) {
    console.error('Error confirming payment:', error);
    res.status(400).json({ error: error.message });
  }
});
```

### 3. Webhook Handler (Recommended)
**Endpoint:** `POST /webhook/stripe`

```javascript
const endpointSecret = "whsec_..."; // Your webhook secret from Stripe Dashboard

app.post('/webhook/stripe', express.raw({type: 'application/json'}), (req, res) => {
  const sig = req.headers['stripe-signature'];
  let event;

  try {
    event = stripe.webhooks.constructEvent(req.body, sig, endpointSecret);
  } catch (err) {
    console.log(`Webhook signature verification failed.`, err.message);
    return res.status(400).send(`Webhook Error: ${err.message}`);
  }

  // Handle the event
  switch (event.type) {
    case 'payment_intent.succeeded':
      const paymentIntent = event.data.object;
      console.log('Payment succeeded:', paymentIntent.id);
      
      // Add credits to user account
      const credits = Math.round(paymentIntent.amount / 100);
      // await addCreditsToUser(paymentIntent.metadata.user_id, credits);
      
      break;
    case 'payment_intent.payment_failed':
      console.log('Payment failed:', event.data.object.id);
      break;
    default:
      console.log(`Unhandled event type ${event.type}`);
  }

  res.json({received: true});
});
```

## Environment Variables
Add these to your `.env` file:

```bash
STRIPE_SECRET_KEY=sk_live_your_stripe_secret_key_here
STRIPE_WEBHOOK_SECRET=whsec_... # Get this from Stripe Dashboard
```

## Frontend Integration
Once your backend is ready, update the frontend to use real endpoints:

```typescript
// In src/hooks/useStripePayment.ts
import { useStripeService } from '../services/stripeService'; // Instead of mock service
```

## Testing
1. Use Stripe test cards for testing:
   - Success: `4242424242424242`
   - Decline: `4000000000000002`
   - 3D Secure: `4000002500003155`

2. Monitor payments in Stripe Dashboard

## Security Notes
- Never expose your secret key on the frontend
- Always validate amounts on the server
- Use webhooks for reliable payment confirmation
- Implement proper error handling
- Log all transactions for audit purposes

## Current Status
- ✅ Frontend integration complete
- ✅ Mock service working for testing
- ⏳ Backend endpoints needed
- ⏳ Webhook setup needed

## Next Steps
1. Implement the backend endpoints above
2. Test with Stripe test cards
3. Set up webhooks in Stripe Dashboard
4. Replace mock service with real service
5. Deploy and test in production