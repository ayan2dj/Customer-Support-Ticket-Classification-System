
from src.model import create_models

# Create models
models = create_models()

# Display available models
print("AVAILABLE MACHINE LEARNING MODELS")

for name, model in models.items():

    print("\nModel Name:", name)
    print(model)

print("\nBoth models created successfully!")
