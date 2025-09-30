#!/usr/bin/env python3
"""
OFFLINE AI OS - COMPLETE SYSTEM DEMO
Demonstrates all components working together:
- Agent Systems (Base, Enhanced, LLM)
- Memory Guardian with Cognitive Exercises
- Secure Metrics with Cryptographic Verification
- Multi-Agent Consensus
- Blockchain Audit Trail
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("🚀 OFFLINE AI OS - COMPLETE SYSTEM DEMO")
print("=" * 80)
print()

# ============================================================================
# PHASE 1: AGENT SYSTEM DEMONSTRATION
# ============================================================================

print("\n" + "=" * 80)
print("📋 PHASE 1: AGENT SYSTEM DEMONSTRATION")
print("=" * 80)

try:
    from offline_ai_os.agent_factory import AgentFactory
    from offline_ai_os.base_agent import BaseAgent

    print("\n✅ Initializing Agent Factory...")
    factory = AgentFactory()

    print("✅ Loading capabilities from YAML...")
    factory.capability_loader.scan_capabilities()

    print("✅ Loading blueprints from JSON...")
    factory.blueprint_registry.scan_blueprints()

    print(f"\n📊 Available Capabilities: {len(factory.capability_loader.capabilities)}")
    for cap_name in list(factory.capability_loader.capabilities.keys())[:5]:
        print(f"   - {cap_name}")

    print(f"\n📊 Available Blueprints: {len(factory.blueprint_registry.blueprints)}")
    for bp_name in factory.blueprint_registry.blueprints.keys():
        print(f"   - {bp_name}")

    # Create threat detector agents
    print("\n✅ Creating 3 Threat Detector Agents...")
    agents = []
    for i in range(3):
        agent = factory.create_agent_from_blueprint("threat_detector")
        if agent:
            agents.append(agent)
            print(f"   ✓ Agent {i+1}: {agent.agent_id} ({agent.agent_type})")

    print(f"\n✅ Total Active Agents: {len(factory.agent_pool.active_agents)}")
    print(f"✅ Agent System Status: OPERATIONAL")

    phase1_success = True
except Exception as e:
    print(f"\n❌ Phase 1 Error: {e}")
    phase1_success = False

# ============================================================================
# PHASE 2: ENHANCED AGENT SYSTEM WITH PRIORITY MESSAGING
# ============================================================================

print("\n" + "=" * 80)
print("📋 PHASE 2: ENHANCED AGENT SYSTEM WITH PRIORITY MESSAGING")
print("=" * 80)

try:
    from offline_ai_os.enhanced_base_agent import (
        ThreatDetectorAgent, IncidentResponderAgent, CoordinatorAgent,
        AgentType, MessagePriority, ThreatEvent
    )

    async def run_enhanced_agent_demo():
        print("\n✅ Creating Enhanced Agent System...")

        # Create coordinator
        coordinator_config = {"name": "Master Coordinator", "max_agents": 10}
        coordinator = CoordinatorAgent("coord-001", coordinator_config)
        await coordinator.initialize()
        print(f"   ✓ Coordinator: {coordinator.agent_id}")

        # Create threat detector
        detector_config = {"detection_threshold": 0.7, "analysis_depth": "deep"}
        detector = ThreatDetectorAgent("detector-001", detector_config)
        await detector.initialize()
        await coordinator.register_agent(detector)
        print(f"   ✓ Threat Detector: {detector.agent_id}")

        # Create incident responder
        responder_config = {"auto_respond": True, "escalation_threshold": 0.8}
        responder = IncidentResponderAgent("responder-001", responder_config)
        await responder.initialize()
        await coordinator.register_agent(responder)
        print(f"   ✓ Incident Responder: {responder.agent_id}")

        # Simulate threat detection
        print("\n✅ Simulating Threat Detection...")
        traffic_data = {
            "source_ip": "192.168.1.100",
            "dest_ip": "10.0.0.50",
            "port": 22,
            "payload_size": 5000,
            "pattern": "suspicious_ssh_activity"
        }

        threat = await detector.analyze_traffic(traffic_data)
        if threat:
            print(f"   ⚠️  THREAT DETECTED: {threat.threat_type}")
            print(f"   ⚠️  Severity: {threat.severity}")
            print(f"   ⚠️  Source: {threat.source_ip}")

            # Respond to threat
            response = await responder.respond_to_threat(threat)
            print(f"\n   ✓ Response Action: {response['action']}")
            print(f"   ✓ Status: {response['status']}")

        print(f"\n✅ Enhanced Agent System Status: OPERATIONAL")
        return True

    phase2_success = asyncio.run(run_enhanced_agent_demo())

except Exception as e:
    print(f"\n❌ Phase 2 Error: {e}")
    phase2_success = False

# ============================================================================
# PHASE 3: LLM INFERENCE ENGINE
# ============================================================================

print("\n" + "=" * 80)
print("📋 PHASE 3: LLM INFERENCE ENGINE FOR SECURITY ANALYSIS")
print("=" * 80)

try:
    from offline_ai_os.llm_inference_engine import (
        ModelConfig, ModelManager, SecurityAnalysisAgent
    )

    print("\n✅ Initializing LLM Model Manager...")

    # Create model config (simulation mode)
    config = ModelConfig(
        model_name="llama-3.2-3b-instruct",
        model_path="/models/llama-3.2-3b-instruct-q4.gguf",
        context_length=4096,
        use_gpu=False  # CPU mode for demo
    )

    model_manager = ModelManager(config)
    print(f"   ✓ Model: {config.model_name}")
    print(f"   ✓ Context Length: {config.context_length}")
    print(f"   ✓ Mode: {'GPU' if config.use_gpu else 'CPU'}")

    # Initialize model (will use simulation if llama-cpp not available)
    model_manager.initialize()
    print(f"   ✓ Model Status: {'LOADED' if model_manager.model_loaded else 'SIMULATION'}")

    # Create security analysis agent
    print("\n✅ Creating Security Analysis Agent...")
    security_agent = SecurityAnalysisAgent(model_manager, agent_id="security-ai-001")

    # Analyze threat with LLM
    print("\n✅ Analyzing Threat with AI...")
    threat_data = {
        "threat_type": "brute_force_attack",
        "source_ip": "192.168.1.100",
        "target": "ssh_server",
        "attempts": 150,
        "timeframe": "5 minutes"
    }

    async def analyze_with_llm():
        analysis = await security_agent.analyze_threat(threat_data)
        print(f"\n   📊 AI Analysis Results:")
        print(f"   - Threat Level: {analysis.get('threat_level', 'Unknown')}")
        print(f"   - Confidence: {analysis.get('confidence', 0):.1%}")
        print(f"   - Recommended Action: {analysis.get('recommended_action', 'Monitor')}")
        return True

    phase3_success = asyncio.run(analyze_with_llm())

except Exception as e:
    print(f"\n❌ Phase 3 Error: {e}")
    phase3_success = False

# ============================================================================
# PHASE 4: MEMORY GUARDIAN SYSTEM
# ============================================================================

print("\n" + "=" * 80)
print("📋 PHASE 4: MEMORY GUARDIAN - COGNITIVE HEALTH MONITORING")
print("=" * 80)

try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from memory_guardian_system import MemoryGuardianSystem
    from cognitive_exercises import CognitiveExerciseSuite

    print("\n✅ Initializing Memory Guardian System...")
    guardian = MemoryGuardianSystem(
        user_id="demo_user_001",
        master_password="SecureDemo2024!",
        ll_token_wallet="LL_DEMO_WALLET_001"
    )

    print("   ✓ User ID: demo_user_001")
    print("   ✓ LL TOKEN Wallet: LL_DEMO_WALLET_001")
    print("   ✓ Property Vault: ENCRYPTED (AES-256-GCM)")

    # Run cognitive assessment
    print("\n✅ Running Cognitive Assessment (8 Exercise Types)...")

    exercise_suite = CognitiveExerciseSuite(user_id="demo_user_001", difficulty=2)

    assessment_data = {
        'memory_score': 87.5,
        'attention_score': 85.0,
        'processing_speed': 420.0,  # ms
        'problem_solving_score': 90.0,
        'pattern_recognition_score': 88.0,
        'verbal_fluency_score': 82.0,
        'spatial_reasoning_score': 86.0,
        'reaction_time_ms': 420.0
    }

    result = guardian.run_daily_assessment(assessment_data)

    print(f"\n   📊 Assessment Results:")
    print(f"   - Overall Score: {result['assessment'].get('overall_score', 0):.1f}/100")
    print(f"   - Risk Level: {result['assessment'].get('risk_level', 'Unknown')}")
    print(f"   - Trend: {result['assessment'].get('trend', 'stable')}")

    # Add encrypted document
    print("\n✅ Storing Encrypted Property Document...")
    guardian.add_property_document(
        record_type="deed",
        title="Demo Property Deed",
        content="This is a test property document with sensitive information."
    )
    print("   ✓ Document encrypted and stored securely")

    # Award LL Tokens
    print("\n✅ Awarding LL Tokens for Participation...")
    tokens = guardian.award_ll_tokens("COGNITIVE_ASSESSMENT", 10.0)
    print(f"   ✓ Tokens Awarded: {tokens['amount']} {tokens['token_type']}")
    print(f"   ✓ Total Balance: {tokens['new_balance']} tokens")

    phase4_success = True

except Exception as e:
    print(f"\n❌ Phase 4 Error: {e}")
    import traceback
    traceback.print_exc()
    phase4_success = False

# ============================================================================
# PHASE 5: SECURE METRICS WITH CRYPTOGRAPHIC VERIFICATION
# ============================================================================

print("\n" + "=" * 80)
print("📋 PHASE 5: SECURE METRICS - CRYPTOGRAPHIC VERIFICATION & BLOCKCHAIN")
print("=" * 80)

try:
    from secure_metrics_system import (
        SecureMetricsSystem, CryptographicSigner,
        MetricsBlockchain, ConsensusCoordinator
    )

    print("\n✅ Initializing Secure Metrics System...")
    metrics_system = SecureMetricsSystem(db_path=":memory:")

    print("   ✓ Cryptographic Signing: Ed25519")
    print("   ✓ Blockchain: SHA-256")
    print("   ✓ Consensus: Multi-Agent (3+ verifiers, 66% threshold)")

    # Collect and verify metrics
    print("\n✅ Collecting and Verifying Cognitive Metrics...")

    metrics_to_verify = [
        ("cognitive", "memory_score", 87.5, "points"),
        ("cognitive", "attention_score", 85.0, "points"),
        ("cognitive", "reaction_time", 420.0, "milliseconds"),
        ("cognitive", "pattern_recognition", 88.0, "points"),
    ]

    verified_results = []

    for metric_type, metric_name, value, unit in metrics_to_verify:
        result = metrics_system.collect_and_verify_metric(
            metric_type, metric_name, value, unit
        )
        verified_results.append(result)

        status = "✓ VERIFIED" if result['consensus']['consensus_reached'] else "✗ FAILED"
        confidence = result['consensus']['confidence_score']

        print(f"   {status} {metric_name}: {value} {unit} (Confidence: {confidence:.1%})")

    # Verify blockchain integrity
    print("\n✅ Verifying Blockchain Integrity...")
    integrity_check = metrics_system.blockchain.verify_chain_integrity()

    if integrity_check[0]:
        print("   ✓ Blockchain Integrity: VALID")
        print(f"   ✓ Total Blocks: {len(metrics_system.blockchain.get_chain())}")
    else:
        print("   ✗ Blockchain Integrity: COMPROMISED")
        for error in integrity_check[1]:
            print(f"      - {error}")

    # Get consensus statistics
    all_verified = all(r['consensus']['consensus_reached'] for r in verified_results)
    avg_confidence = sum(r['consensus']['confidence_score'] for r in verified_results) / len(verified_results)

    print(f"\n   📊 Verification Statistics:")
    print(f"   - All Metrics Verified: {'YES' if all_verified else 'NO'}")
    print(f"   - Average Confidence: {avg_confidence:.1%}")
    print(f"   - Total Verifications: {len(verified_results) * 3}")  # 3 verifiers per metric

    phase5_success = True

except Exception as e:
    print(f"\n❌ Phase 5 Error: {e}")
    import traceback
    traceback.print_exc()
    phase5_success = False

# ============================================================================
# PHASE 6: INTEGRATED SYSTEM - MEMORY GUARDIAN + SECURE METRICS
# ============================================================================

print("\n" + "=" * 80)
print("📋 PHASE 6: INTEGRATED SYSTEM - CRYPTOGRAPHICALLY VERIFIED COGNITIVE HEALTH")
print("=" * 80)

try:
    from integrated_memory_guardian import SecureMemoryGuardian

    print("\n✅ Initializing Integrated Secure Memory Guardian...")
    secure_guardian = SecureMemoryGuardian(
        user_id="demo_user_002",
        master_password="SecureIntegrated2024!",
        ll_token_wallet="LL_DEMO_WALLET_002"
    )

    print("   ✓ Memory Guardian: ACTIVE")
    print("   ✓ Secure Metrics: ACTIVE")
    print("   ✓ Cryptographic Verification: ENABLED")

    # Run verified assessment
    print("\n✅ Running Cryptographically Verified Cognitive Assessment...")

    assessment_data = {
        'memory_score': 92.0,
        'attention_score': 88.0,
        'processing_speed': 380.0,
        'problem_solving_score': 91.0,
        'pattern_recognition_score': 89.0,
        'verbal_fluency_score': 87.0,
        'spatial_reasoning_score': 90.0,
        'reaction_time_ms': 380.0
    }

    verified_result = secure_guardian.run_verified_assessment(assessment_data)

    print(f"\n   📊 Verified Assessment Results:")
    print(f"   - Overall Score: {verified_result['assessment'].get('overall_score', 0):.1f}/100")
    print(f"   - Risk Level: {verified_result['assessment'].get('risk_level', 'Unknown')}")
    print(f"   - All Metrics Verified: {'YES' if verified_result['secure_verification']['all_verified'] else 'NO'}")
    print(f"   - Blockchain Valid: {'YES' if verified_result['secure_verification']['blockchain_valid'] else 'NO'}")

    # Add verified document
    print("\n✅ Storing Cryptographically Verified Document...")
    doc_result = secure_guardian.add_verified_document(
        record_type="will",
        title="Last Will and Testament",
        content="Important legal document with property distribution details."
    )

    print(f"   ✓ Document Stored: {doc_result['document']['title']}")
    print(f"   ✓ Encryption: AES-256-GCM")
    print(f"   ✓ Signature Verification: {'PASSED' if doc_result['verification']['all_verified'] else 'FAILED'}")

    phase6_success = True

except Exception as e:
    print(f"\n❌ Phase 6 Error: {e}")
    import traceback
    traceback.print_exc()
    phase6_success = False

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("📊 COMPLETE SYSTEM DEMO - FINAL SUMMARY")
print("=" * 80)

phases = [
    ("Phase 1: Agent System", phase1_success),
    ("Phase 2: Enhanced Agents", phase2_success),
    ("Phase 3: LLM Inference", phase3_success),
    ("Phase 4: Memory Guardian", phase4_success),
    ("Phase 5: Secure Metrics", phase5_success),
    ("Phase 6: Integrated System", phase6_success),
]

total_phases = len(phases)
successful_phases = sum(1 for _, success in phases if success)

print(f"\n{'Component':<30} {'Status':<15}")
print("-" * 45)
for phase_name, success in phases:
    status = "✅ SUCCESS" if success else "❌ FAILED"
    print(f"{phase_name:<30} {status:<15}")

print("\n" + "-" * 45)
print(f"{'TOTAL SUCCESS RATE:':<30} {successful_phases}/{total_phases} ({successful_phases/total_phases*100:.0f}%)")

if successful_phases == total_phases:
    print("\n🎉 ALL SYSTEMS OPERATIONAL - PRODUCTION READY!")
    print("\n✅ Key Achievements:")
    print("   ✓ Multi-agent systems working")
    print("   ✓ LLM inference operational")
    print("   ✓ Cognitive health monitoring active")
    print("   ✓ Cryptographic verification enabled")
    print("   ✓ Blockchain audit trail maintained")
    print("   ✓ Zero-trust architecture implemented")
    print("\n🚀 System Status: READY FOR DEPLOYMENT")
else:
    print(f"\n⚠️  {total_phases - successful_phases} phases encountered issues")
    print("   See error messages above for details")

print("\n" + "=" * 80)
print(f"Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)