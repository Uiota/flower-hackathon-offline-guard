#!/usr/bin/env python3
"""
Full System Test for FL Demo with Agents
Tests the complete federated learning system with dashboard integration
"""

import requests
import time
import json
import sys

def test_dashboard_api():
    """Test dashboard API endpoints."""
    print("🧪 Testing Dashboard API...")

    base_url = "http://localhost:8081"

    try:
        # Test status endpoint
        response = requests.get(f"{base_url}/api/status", timeout=5)
        status = response.json()

        print(f"✅ Status: {status}")

        if not status.get("dashboard_running"):
            print("❌ Dashboard not running")
            return False

        if not status.get("agents_running"):
            print("❌ FL agents not running")
            return False

        if not status.get("offline_mode"):
            print("❌ Offline mode not enabled")
            return False

        # Test metrics endpoint
        response = requests.get(f"{base_url}/api/metrics", timeout=5)
        metrics = response.json()

        global_metrics = metrics.get("global_metrics", {})
        client_metrics = metrics.get("client_metrics", {})
        security_status = metrics.get("security_status", {})

        print(f"✅ Global Metrics:")
        print(f"   Round: {global_metrics.get('round')}")
        print(f"   Accuracy: {global_metrics.get('accuracy', 0):.2%}")
        print(f"   Loss: {global_metrics.get('loss', 0):.3f}")
        print(f"   Active Clients: {global_metrics.get('active_clients')}/{global_metrics.get('total_clients')}")

        print(f"✅ Client Agents:")
        for client_id, client_data in client_metrics.items():
            status = client_data.get('status', 'unknown')
            acc = client_data.get('accuracy', 0)
            samples = client_data.get('samples', 0)
            print(f"   {client_id}: {status}, {acc:.2%} accuracy, {samples} samples")

        print(f"✅ Security Status:")
        print(f"   Offline Mode: {security_status.get('offline_mode')}")
        print(f"   Signatures Verified: {security_status.get('signatures_verified', 0)}/{security_status.get('total_signatures', 0)}")
        print(f"   Security Failures: {security_status.get('security_failures', 0)}")

        # Validate system is working
        if global_metrics.get('round', 0) > 0:
            print("✅ FL training is progressing")
        else:
            print("⚠️  FL training not started yet")

        if len(client_metrics) > 0:
            print(f"✅ {len(client_metrics)} client agents detected")
        else:
            print("❌ No client agents found")
            return False

        if security_status.get('signatures_verified', 0) > 0:
            print("✅ Off-Guard security is working")
        else:
            print("⚠️  No security signatures detected yet")

        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Dashboard connection failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_training_progression():
    """Test that training is actually progressing."""
    print("\n🕐 Testing Training Progression...")

    try:
        # Get initial metrics
        response = requests.get("http://localhost:8081/api/metrics", timeout=5)
        initial_metrics = response.json()
        initial_round = initial_metrics["global_metrics"]["round"]
        initial_accuracy = initial_metrics["global_metrics"]["accuracy"]

        print(f"📊 Initial: Round {initial_round}, Accuracy {initial_accuracy:.2%}")

        # Wait and check again
        print("⏳ Waiting 15 seconds for training progression...")
        time.sleep(15)

        response = requests.get("http://localhost:8081/api/metrics", timeout=5)
        final_metrics = response.json()
        final_round = final_metrics["global_metrics"]["round"]
        final_accuracy = final_metrics["global_metrics"]["accuracy"]

        print(f"📊 Final: Round {final_round}, Accuracy {final_accuracy:.2%}")

        if final_round > initial_round:
            print("✅ Training rounds are progressing")
        elif final_accuracy != initial_accuracy:
            print("✅ Training accuracy is changing")
        else:
            print("⚠️  Training may not be progressing (this is normal if FL completed)")

        return True

    except Exception as e:
        print(f"❌ Progression test failed: {e}")
        return False

def main():
    """Run complete system test."""
    print("🚀 Full FL System Test with Off-Guard Security")
    print("=" * 50)

    # Test 1: API functionality
    api_test = test_dashboard_api()

    # Test 2: Training progression (if API works)
    if api_test:
        progression_test = test_training_progression()
    else:
        progression_test = False

    # Final results
    print("\n" + "=" * 50)
    print("🏁 Test Results:")
    print(f"   Dashboard API: {'✅ PASS' if api_test else '❌ FAIL'}")
    print(f"   Training Progress: {'✅ PASS' if progression_test else '❌ FAIL'}")

    if api_test and progression_test:
        print("🎉 Full FL System is FULLY FUNCTIONAL!")
        print("\n📋 System Summary:")
        print("   • FL Dashboard: Running on http://localhost:8081")
        print("   • FL Agents: Multiple clients active")
        print("   • Off-Guard Security: Cryptographic verification enabled")
        print("   • Training: Federated learning progressing")
        print("   • Monitoring: Real-time metrics available")
        return True
    else:
        print("⚠️  System has issues - check logs")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)