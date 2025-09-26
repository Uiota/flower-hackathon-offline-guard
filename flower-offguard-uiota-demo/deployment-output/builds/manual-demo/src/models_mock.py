"""
Mock models module for testing without PyTorch
"""

class MockModel:
    """Mock neural network model for testing."""

    def __init__(self):
        self.parameters = [1.0, 2.0, 3.0]

    def state_dict(self):
        return {"layer1.weight": self.parameters}

    def load_state_dict(self, state_dict):
        self.parameters = state_dict.get("layer1.weight", [1.0, 2.0, 3.0])

    def forward(self, x):
        return sum(self.parameters) * x

def get_model():
    """Get a mock model instance."""
    return MockModel()

def set_model_parameters(model, parameters):
    """Set model parameters."""
    model.parameters = parameters

def get_model_parameters(model):
    """Get model parameters."""
    return model.parameters