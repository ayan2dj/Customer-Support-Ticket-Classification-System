
from src.predict import predict_ticket

# Five new ticket descriptions
test_tickets = [
    "My sign in credentials are not being accepted.",
    "The software crashes whenever I open settings.",
    "I need an export of last month's revenue.",
    "Please change the email linked to my profile.",
    "The dashboard takes several minutes to respond."
]

print("NEW TICKET PREDICTION TESTING")

for i, ticket in enumerate(test_tickets, start=1):

    category, confidence = predict_ticket(ticket)

    print("\nTest Case:", i)
    print("Input:", ticket)
    print("Predicted Category:", category)
    print(f"Confidence: {confidence:.2f}%")
