#!/usr/bin/env python3
"""
Offline LLM Inference Engine
Supports multiple models: Llama, Mistral, Phi, Qwen
Optimized for cybersecurity use cases
"""

import asyncio
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import json
import time

# ==================== MODEL CONFIGURATION ====================

class ModelType(Enum):
    """Supported model types"""
    LLAMA = "llama"
    MISTRAL = "mistral"
    PHI = "phi"
    QWEN = "qwen"
    CODELLAMA = "codellama"

@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    model_type: ModelType
    model_path: str
    context_length: int
    parameters: int  # in billions
    quantization: str  # GGUF, GPTQ, AWQ, FP16
    use_gpu: bool
    gpu_layers: int
    batch_size: int
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048

# ==================== MODEL REGISTRY ====================

# Pre-configured models for offline use
MODEL_REGISTRY = {
    "llama-3.2-3b": ModelConfig(
        name="Llama 3.2 3B Instruct",
        model_type=ModelType.LLAMA,
        model_path="/models/llama-3.2-3b/model.gguf",
        context_length=8192,
        parameters=3,
        quantization="Q4_K_M",
        use_gpu=True,
        gpu_layers=32,
        batch_size=512,
    ),
    "mistral-7b": ModelConfig(
        name="Mistral 7B Instruct",
        model_type=ModelType.MISTRAL,
        model_path="/models/mistral-7b/model.gguf",
        context_length=32768,
        parameters=7,
        quantization="Q5_K_M",
        use_gpu=True,
        gpu_layers=32,
        batch_size=512,
    ),
    "phi-3-mini": ModelConfig(
        name="Phi-3 Mini 4K",
        model_type=ModelType.PHI,
        model_path="/models/phi-3/model.gguf",
        context_length=4096,
        parameters=3.8,
        quantization="Q4_K_M",
        use_gpu=True,
        gpu_layers=32,
        batch_size=256,
    ),
    "codellama-7b": ModelConfig(
        name="CodeLlama 7B",
        model_type=ModelType.CODELLAMA,
        model_path="/models/codellama-7b/model.gguf",
        context_length=16384,
        parameters=7,
        quantization="Q4_K_M",
        use_gpu=True,
        gpu_layers=32,
        batch_size=512,
    ),
}

# ==================== INFERENCE ENGINE ====================

class LLMInferenceEngine:
    """
    Offline LLM inference using llama.cpp
    Supports GGUF quantized models
    """

    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.model = None
        self.loaded = False

    def load_model(self):
        """Load model into memory"""
        print(f"\n[LLM] Loading model: {self.config.name}")
        print(f"  Path: {self.config.model_path}")
        print(f"  Quantization: {self.config.quantization}")
        print(f"  GPU Layers: {self.config.gpu_layers}")

        try:
            from llama_cpp import Llama

            self.model = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.context_length,
                n_batch=self.config.batch_size,
                n_gpu_layers=self.config.gpu_layers if self.config.use_gpu else 0,
                verbose=False
            )

            self.loaded = True
            print(f"[LLM] ✓ Model loaded successfully")

        except ImportError:
            print("[LLM] llama-cpp-python not installed, using simulation mode")
            self.loaded = True  # Allow demo to continue
        except Exception as e:
            print(f"[LLM] Error loading model: {e}")
            print("[LLM] Using simulation mode")
            self.loaded = True

    async def generate(self,
                      prompt: str,
                      max_tokens: Optional[int] = None,
                      temperature: Optional[float] = None,
                      stop: Optional[List[str]] = None) -> str:
        """Generate text completion"""

        if not self.loaded:
            self.load_model()

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        print(f"\n[LLM] Generating response (max {max_tokens} tokens)...")
        start_time = time.time()

        if self.model:
            # Real inference
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                stop=stop or [],
                echo=False
            )

            response = output["choices"][0]["text"]

        else:
            # Simulate for demo
            await asyncio.sleep(1)  # Simulate inference time
            response = self._simulate_response(prompt)

        elapsed = time.time() - start_time
        tokens = len(response.split())
        tokens_per_sec = tokens / elapsed if elapsed > 0 else 0

        print(f"[LLM] ✓ Generated {tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")

        return response

    async def generate_stream(self,
                             prompt: str,
                             max_tokens: Optional[int] = None,
                             temperature: Optional[float] = None) -> AsyncIterator[str]:
        """Generate text with streaming"""

        if not self.loaded:
            self.load_model()

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        if self.model:
            # Real streaming
            stream = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                stream=True
            )

            for output in stream:
                chunk = output["choices"][0]["text"]
                yield chunk
        else:
            # Simulate streaming
            response = self._simulate_response(prompt)
            for word in response.split():
                await asyncio.sleep(0.05)
                yield word + " "

    def _simulate_response(self, prompt: str) -> str:
        """Simulate LLM response for demo"""
        if "threat" in prompt.lower() or "security" in prompt.lower():
            return """Based on the analysis, this appears to be a potential security threat.
The indicators suggest suspicious network behavior with characteristics of:
1. Unusual outbound connections to non-standard ports
2. High volume of data transfer
3. Communication patterns consistent with C2 traffic

Recommended actions:
- Isolate the affected host immediately
- Capture network traffic for forensic analysis
- Check for indicators of compromise (IOCs)
- Scan for malware using multiple engines"""

        elif "malware" in prompt.lower():
            return """Malware analysis results:
- File appears to be packed/obfuscated
- Exhibits suspicious API calls (network, process injection)
- Creates persistence mechanisms
- Communicates with external servers

Classification: Likely trojan or RAT
Confidence: 85%
Recommended: Immediate quarantine and full system scan"""

        else:
            return "Analysis complete. The system appears normal with no immediate threats detected."

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.config.name,
            "type": self.config.model_type.value,
            "parameters": f"{self.config.parameters}B",
            "context_length": self.config.context_length,
            "quantization": self.config.quantization,
            "loaded": self.loaded,
            "gpu_enabled": self.config.use_gpu
        }

# ==================== MULTI-MODEL MANAGER ====================

class MultiModelManager:
    """
    Manages multiple LLM models
    Routes requests to appropriate models
    """

    def __init__(self):
        self.engines: Dict[str, LLMInferenceEngine] = {}
        self.active_model: Optional[str] = None

    def register_model(self, model_id: str, config: ModelConfig):
        """Register a model"""
        engine = LLMInferenceEngine(config)
        self.engines[model_id] = engine

        if not self.active_model:
            self.active_model = model_id

        print(f"[MultiModel] Registered: {model_id}")

    def load_model(self, model_id: str):
        """Load specific model"""
        if model_id not in self.engines:
            raise ValueError(f"Model {model_id} not registered")

        self.engines[model_id].load_model()
        self.active_model = model_id

    def switch_model(self, model_id: str):
        """Switch active model"""
        if model_id not in self.engines:
            raise ValueError(f"Model {model_id} not registered")

        self.active_model = model_id
        print(f"[MultiModel] Switched to: {model_id}")

    async def generate(self,
                      prompt: str,
                      model_id: Optional[str] = None,
                      **kwargs) -> str:
        """Generate using specified or active model"""
        model_id = model_id or self.active_model

        if not model_id or model_id not in self.engines:
            raise ValueError("No active model")

        return await self.engines[model_id].generate(prompt, **kwargs)

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        return [
            {
                "id": model_id,
                **engine.get_model_info(),
                "active": model_id == self.active_model
            }
            for model_id, engine in self.engines.items()
        ]

# ==================== PROMPT TEMPLATES ====================

class PromptTemplates:
    """Cybersecurity-specific prompt templates"""

    @staticmethod
    def threat_analysis(event_data: Dict[str, Any]) -> str:
        """Template for threat analysis"""
        return f"""<|system|>
You are a cybersecurity expert analyzing security events. Provide detailed threat analysis.
</s>
<|user|>
Analyze this security event and determine if it's a threat:

Event Data:
{json.dumps(event_data, indent=2)}

Provide:
1. Threat assessment (benign/suspicious/malicious)
2. Confidence level (0-100%)
3. Attack type if malicious
4. Recommended actions
5. IOCs to monitor
</s>
<|assistant|>
"""

    @staticmethod
    def malware_classification(file_info: Dict[str, Any]) -> str:
        """Template for malware classification"""
        return f"""<|system|>
You are a malware analyst. Classify malware based on behavioral indicators.
</s>
<|user|>
Classify this potential malware:

File Information:
{json.dumps(file_info, indent=2)}

Provide:
1. Malware family
2. Threat type (trojan/ransomware/worm/etc)
3. Capabilities
4. Severity (critical/high/medium/low)
5. Mitigation steps
</s>
<|assistant|>
"""

    @staticmethod
    def incident_response(incident: Dict[str, Any]) -> str:
        """Template for incident response recommendations"""
        return f"""<|system|>
You are an incident response expert. Provide step-by-step response procedures.
</s>
<|user|>
Security incident detected:

Incident Details:
{json.dumps(incident, indent=2)}

Provide incident response plan:
1. Immediate containment actions
2. Investigation steps
3. Eradication procedures
4. Recovery steps
5. Post-incident activities
</s>
<|assistant|>
"""

    @staticmethod
    def log_analysis(logs: List[str]) -> str:
        """Template for log analysis"""
        return f"""<|system|>
You are a security analyst examining system logs for anomalies.
</s>
<|user|>
Analyze these logs and identify any security concerns:

Logs:
{chr(10).join(logs[:20])}  # Limit to 20 lines

Identify:
1. Suspicious patterns
2. Potential security events
3. Anomalies requiring investigation
4. Priority items
</s>
<|assistant|>
"""

    @staticmethod
    def code_vulnerability_scan(code: str) -> str:
        """Template for code security analysis"""
        return f"""<|system|>
You are a security code reviewer. Identify vulnerabilities and security issues.
</s>
<|user|>
Review this code for security vulnerabilities:

```
{code}
```

Identify:
1. Security vulnerabilities
2. Severity of each issue
3. Exploitation scenarios
4. Remediation recommendations
</s>
<|assistant|>
"""

# ==================== SECURITY-SPECIFIC LLM AGENT ====================

class SecurityAnalysisAgent:
    """
    AI agent specialized for security analysis using LLM
    """

    def __init__(self, model_manager: MultiModelManager):
        self.model_manager = model_manager
        self.templates = PromptTemplates()

    async def analyze_threat(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze security event with LLM"""
        prompt = self.templates.threat_analysis(event_data)

        response = await self.model_manager.generate(
            prompt,
            max_tokens=1024,
            temperature=0.3  # Lower temp for more focused analysis
        )

        return self._parse_threat_response(response)

    async def classify_malware(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Classify malware using LLM"""
        prompt = self.templates.malware_classification(file_info)

        response = await self.model_manager.generate(
            prompt,
            max_tokens=1024,
            temperature=0.3
        )

        return self._parse_malware_response(response)

    async def generate_incident_response(self, incident: Dict[str, Any]) -> str:
        """Generate incident response plan"""
        prompt = self.templates.incident_response(incident)

        response = await self.model_manager.generate(
            prompt,
            max_tokens=2048,
            temperature=0.4
        )

        return response

    async def analyze_logs(self, logs: List[str]) -> Dict[str, Any]:
        """Analyze system logs"""
        prompt = self.templates.log_analysis(logs)

        response = await self.model_manager.generate(
            prompt,
            max_tokens=1536,
            temperature=0.3
        )

        return {"analysis": response, "log_count": len(logs)}

    def _parse_threat_response(self, response: str) -> Dict[str, Any]:
        """Parse threat analysis response"""
        # Simplified parsing - in production, use structured output
        return {
            "analysis": response,
            "is_threat": "malicious" in response.lower(),
            "confidence": 0.85  # Would extract from response
        }

    def _parse_malware_response(self, response: str) -> Dict[str, Any]:
        """Parse malware classification response"""
        return {
            "classification": response,
            "family": "unknown",  # Would extract from response
            "severity": "high" if "critical" in response.lower() else "medium"
        }

# ==================== DEMO ====================

async def demo_llm_inference():
    """Demonstrate LLM inference for security"""
    print("\n" + "="*70)
    print("Offline LLM Inference Engine Demo")
    print("="*70)

    # Initialize multi-model manager
    manager = MultiModelManager()

    # Register models
    manager.register_model("llama-3b", MODEL_REGISTRY["llama-3.2-3b"])
    manager.register_model("mistral-7b", MODEL_REGISTRY["mistral-7b"])

    # Load default model
    print("\n→ Loading default model...")
    manager.load_model("llama-3b")

    # List available models
    print("\n→ Available Models:")
    for model in manager.list_models():
        active = " [ACTIVE]" if model["active"] else ""
        print(f"  • {model['id']}: {model['name']} ({model['parameters']}){active}")

    # Create security agent
    security_agent = SecurityAnalysisAgent(manager)

    # Demo 1: Threat Analysis
    print("\n" + "-"*70)
    print("Demo 1: Threat Analysis")
    print("-"*70)

    threat_event = {
        "source_ip": "192.168.1.100",
        "destination_ip": "10.0.0.50",
        "destination_port": 4444,
        "protocol": "TCP",
        "bytes_transferred": 50000,
        "packets": 250,
        "flags": ["SYN", "ACK", "PSH"],
        "duration": 300
    }

    result = await security_agent.analyze_threat(threat_event)
    print("\nThreat Analysis Result:")
    print(result["analysis"][:500] + "..." if len(result["analysis"]) > 500 else result["analysis"])

    # Demo 2: Malware Classification
    print("\n" + "-"*70)
    print("Demo 2: Malware Classification")
    print("-"*70)

    malware_info = {
        "filename": "suspicious.exe",
        "size": 1024000,
        "api_calls": ["CreateProcess", "WriteProcessMemory", "VirtualAllocEx"],
        "network_activity": True,
        "persistence": ["Registry Run key"],
        "packed": True
    }

    classification = await security_agent.classify_malware(malware_info)
    print("\nMalware Classification:")
    print(classification["classification"][:500] + "..." if len(classification["classification"]) > 500 else classification["classification"])

    # Demo 3: Incident Response
    print("\n" + "-"*70)
    print("Demo 3: Incident Response Plan")
    print("-"*70)

    incident = {
        "type": "ransomware",
        "affected_systems": 5,
        "encryption_started": "2025-01-15 14:30:00",
        "ransom_note": True,
        "network_isolation": False
    }

    response_plan = await security_agent.generate_incident_response(incident)
    print("\nIncident Response Plan:")
    print(response_plan[:500] + "..." if len(response_plan) > 500 else response_plan)

    print("\n" + "="*70)
    print("LLM Inference Demo Complete!")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(demo_llm_inference())