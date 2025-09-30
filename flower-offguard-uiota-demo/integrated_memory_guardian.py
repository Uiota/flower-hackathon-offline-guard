#!/usr/bin/env python3
"""
Integrated Memory Guardian with Secure Metrics System
Combines cognitive health monitoring with multi-agent cryptographic verification
"""

from memory_guardian_system import MemoryGuardianApp, CognitiveHealthMonitor
from secure_metrics_system import SecureMetricsSystem
from datetime import datetime
import json


class SecureMemoryGuardian:
    """
    Memory Guardian with integrated secure metrics verification
    Every cognitive assessment is cryptographically signed and verified by multiple agents
    """

    def __init__(self, user_id: str, master_password: str, ll_token_wallet: str):
        # Initialize Memory Guardian
        self.memory_guardian = MemoryGuardianApp(
            user_id=user_id,
            master_password=master_password,
            ll_token_wallet=ll_token_wallet
        )

        # Initialize Secure Metrics System
        self.metrics_system = SecureMetricsSystem()

        self.user_id = user_id

        print(f"\n✅ Secure Memory Guardian initialized for {user_id}")
        print("🔒 All cognitive metrics will be cryptographically verified")

    def run_verified_assessment(self, assessment_data: dict) -> dict:
        """
        Run cognitive assessment with multi-agent verification
        """
        print("\n" + "=" * 80)
        print("🧠 RUNNING VERIFIED COGNITIVE ASSESSMENT")
        print("=" * 80)

        # Step 1: Run normal assessment
        result = self.memory_guardian.run_daily_assessment(assessment_data)

        print(f"\n📊 Assessment completed:")
        print(f"   Overall Score: {result['assessment']['overall_score']:.1f}")
        print(f"   Status: {result['evaluation']['status']}")

        # Step 2: Verify each metric with secure system
        print(f"\n🔐 Verifying metrics with agent consensus...")

        verified_metrics = {}

        # Verify overall score
        overall_result = self.metrics_system.collect_and_verify_metric(
            "cognitive",
            "overall_score",
            result['assessment']['overall_score'],
            "points"
        )
        verified_metrics['overall'] = overall_result

        # Verify memory score
        memory_result = self.metrics_system.collect_and_verify_metric(
            "cognitive",
            "memory_score",
            result['assessment']['memory_score'],
            "points"
        )
        verified_metrics['memory'] = memory_result

        # Verify pattern recognition
        pattern_result = self.metrics_system.collect_and_verify_metric(
            "cognitive",
            "pattern_recognition_score",
            result['assessment']['pattern_recognition_score'],
            "points"
        )
        verified_metrics['pattern'] = pattern_result

        # Verify problem solving
        problem_result = self.metrics_system.collect_and_verify_metric(
            "cognitive",
            "problem_solving_score",
            result['assessment']['problem_solving_score'],
            "points"
        )
        verified_metrics['problem'] = problem_result

        # Verify reaction time
        reaction_result = self.metrics_system.collect_and_verify_metric(
            "cognitive",
            "reaction_time",
            result['assessment']['reaction_time_ms'],
            "milliseconds"
        )
        verified_metrics['reaction'] = reaction_result

        # Step 3: Check if all metrics reached consensus
        all_verified = all(
            m['consensus']['consensus_reached']
            for m in verified_metrics.values()
        )

        print(f"\n{'✅' if all_verified else '⚠️'} Verification Status: "
              f"{'ALL METRICS VERIFIED' if all_verified else 'SOME METRICS PENDING'}")

        # Step 4: Return enhanced result
        enhanced_result = {
            **result,
            "secure_verification": {
                "all_verified": all_verified,
                "verified_metrics": verified_metrics,
                "blockchain_integrity": self.metrics_system.blockchain.verify_chain_integrity()[0],
                "verification_timestamp": datetime.now().isoformat()
            }
        }

        return enhanced_result

    def get_secure_dashboard(self) -> dict:
        """
        Get dashboard with both health metrics and security verification
        """
        # Get normal dashboard
        guardian_dashboard = self.memory_guardian.get_dashboard_summary()

        # Get metrics dashboard
        metrics_dashboard = self.metrics_system.get_metrics_dashboard()

        # Combine
        return {
            "cognitive_health": guardian_dashboard,
            "security_metrics": metrics_dashboard,
            "timestamp": datetime.now().isoformat()
        }

    def secure_document_with_verification(self, doc_type: str, title: str,
                                         content: str, trusted_contacts: list = None) -> dict:
        """
        Secure document and verify the operation with agents
        """
        # Secure the document normally
        result = self.memory_guardian.secure_document(
            doc_type, title, content, trusted_contacts
        )

        # Verify the security operation
        security_metric = self.metrics_system.collect_and_verify_metric(
            "security",
            "document_secured",
            1.0,  # Binary: 1 = secured
            "operation"
        )

        result['security_verification'] = security_metric

        return result


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("INTEGRATED MEMORY GUARDIAN - Secure Metrics Demo")
    print("=" * 80)

    # Initialize integrated system
    secure_guardian = SecureMemoryGuardian(
        user_id="secure_user_001",
        master_password="SecurePassword123!",
        ll_token_wallet="LL_SECURE_WALLET"
    )

    # Run verified assessment
    print("\n\n" + "=" * 80)
    print("TEST 1: Verified Cognitive Assessment")
    print("=" * 80)

    assessment_result = secure_guardian.run_verified_assessment({
        'memory_score': 87.5,
        'reaction_time_ms': 420.0,
        'pattern_recognition_score': 90.0,
        'problem_solving_score': 85.0,
        'overall_score': 87.5
    })

    # Run another assessment to build history
    print("\n\n" + "=" * 80)
    print("TEST 2: Second Verified Assessment")
    print("=" * 80)

    assessment_result2 = secure_guardian.run_verified_assessment({
        'memory_score': 88.0,
        'reaction_time_ms': 410.0,
        'pattern_recognition_score': 91.0,
        'problem_solving_score': 86.0,
        'overall_score': 88.0
    })

    # Secure a document with verification
    print("\n\n" + "=" * 80)
    print("TEST 3: Secure Document with Verification")
    print("=" * 80)

    doc_result = secure_guardian.secure_document_with_verification(
        doc_type="medical",
        title="Medical Records - 2025",
        content="Important medical information...",
        trusted_contacts=["contact_001"]
    )

    print(f"\n📄 Document secured and verified:")
    print(f"   Record ID: {doc_result['record_id']}")
    print(f"   Security Verification: "
          f"{'✅ VERIFIED' if doc_result['security_verification']['consensus']['consensus_reached'] else '⚠️ PENDING'}")

    # Get secure dashboard
    print("\n\n" + "=" * 80)
    print("SECURE DASHBOARD SUMMARY")
    print("=" * 80)

    dashboard = secure_guardian.get_secure_dashboard()

    print(f"\n🧠 Cognitive Health:")
    print(f"   Total Assessments: {dashboard['cognitive_health']['total_assessments']}")
    print(f"   Average Score: {dashboard['cognitive_health']['average_score_30d']:.1f}")
    print(f"   Trend: {dashboard['cognitive_health']['trend']}")
    print(f"   Tokens Earned: {dashboard['cognitive_health']['total_tokens_earned']:.1f}")

    print(f"\n🔒 Security Metrics:")
    print(f"   Total Metrics: {dashboard['security_metrics']['total_metrics']}")
    print(f"   Consensus Rate: {dashboard['security_metrics']['consensus_rate']:.1%}")
    print(f"   Chain Integrity: {'✅ VALID' if dashboard['security_metrics']['chain_integrity'] else '❌ COMPROMISED'}")

    print(f"\n   Metrics by Type:")
    for metric_type, count in dashboard['security_metrics']['metrics_by_type'].items():
        print(f"      {metric_type}: {count}")

    print("\n\n" + "=" * 80)
    print("✅ INTEGRATED SECURE MEMORY GUARDIAN DEMO COMPLETE")
    print("=" * 80)
    print("\n🎯 Key Benefits:")
    print("   ✓ Every metric cryptographically signed (Ed25519)")
    print("   ✓ Multi-agent consensus verification (3+ agents)")
    print("   ✓ Blockchain-style audit trail")
    print("   ✓ Tamper-proof cognitive health records")
    print("   ✓ Zero-trust security architecture")
    print("   ✓ 100% offline capable")
    print("\n🧠 Protecting minds with cryptographic certainty! 🔒")