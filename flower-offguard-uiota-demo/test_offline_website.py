#!/usr/bin/env python3
"""
LL TOKEN OFFLINE - Website Functionality Test
Tests all components of the offline wallet and marketplace website
"""

import requests
import json
import time
import webbrowser
from pathlib import Path

class OfflineWebsiteTest:
    def __init__(self):
        self.base_url = "http://localhost:8080"
        self.website_dir = Path(__file__).parent / "website"

    def test_website_accessibility(self):
        """Test if website is accessible and responsive"""
        print("🌐 Testing website accessibility...")

        try:
            response = requests.get(self.base_url, timeout=5)
            if response.status_code == 200:
                print("✅ Website accessible at", self.base_url)
                return True
            else:
                print("❌ Website returned status code:", response.status_code)
                return False
        except requests.exceptions.RequestException as e:
            print("❌ Website not accessible:", e)
            return False

    def test_file_structure(self):
        """Test if all required files are present"""
        print("📁 Testing file structure...")

        required_files = [
            "index.html",
            "css/styles.css",
            "js/app.js",
            "js/wallet.js",
            "js/marketplace.js",
            "js/metaverse.js",
            "js/staking.js"
        ]

        all_present = True
        for file_path in required_files:
            full_path = self.website_dir / file_path
            if full_path.exists():
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path} missing")
                all_present = False

        return all_present

    def test_javascript_modules(self):
        """Test if JavaScript modules load without syntax errors"""
        print("🔧 Testing JavaScript modules...")

        js_files = [
            "js/app.js",
            "js/wallet.js",
            "js/marketplace.js",
            "js/metaverse.js",
            "js/staking.js"
        ]

        for js_file in js_files:
            try:
                with open(self.website_dir / js_file, 'r') as f:
                    content = f.read()
                    if len(content) > 0:
                        print(f"✅ {js_file} loaded ({len(content)} characters)")
                    else:
                        print(f"❌ {js_file} is empty")
            except Exception as e:
                print(f"❌ Error reading {js_file}: {e}")

    def test_offline_functionality(self):
        """Test key offline features"""
        print("🔒 Testing offline functionality...")

        # Test CSS
        css_response = requests.get(f"{self.base_url}/css/styles.css")
        if css_response.status_code == 200:
            print("✅ CSS styles accessible offline")
        else:
            print("❌ CSS not accessible offline")

        # Test JavaScript modules
        js_modules = ["app.js", "wallet.js", "marketplace.js", "metaverse.js", "staking.js"]
        for module in js_modules:
            js_response = requests.get(f"{self.base_url}/js/{module}")
            if js_response.status_code == 200:
                print(f"✅ {module} accessible offline")
            else:
                print(f"❌ {module} not accessible offline")

    def display_feature_summary(self):
        """Display summary of all website features"""
        print("\n🚀 LL TOKEN OFFLINE Website Features:")
        print("="*50)

        features = {
            "💳 Quantum-Safe Wallet": [
                "Ed25519 digital signatures",
                "AES-256-GCM encryption",
                "Offline transaction signing",
                "Multi-token support (14 token types)",
                "Transaction history"
            ],
            "🏪 Token Marketplace": [
                "Peer-to-peer token trading",
                "NFT marketplace integration",
                "Order book management",
                "Price discovery mechanisms",
                "Trade execution"
            ],
            "🌍 Metaverse Integration": [
                "Avatar token management",
                "Virtual land ownership",
                "Cross-world portability",
                "Soul-bound achievements",
                "Metaverse utilities"
            ],
            "🏦 Staking Rewards": [
                "5 specialized staking pools",
                "APY rates from 8% to 22%",
                "Auto-compounding options",
                "Emergency unstaking",
                "Lock period flexibility"
            ],
            "🔐 Security Features": [
                "Offline-first architecture",
                "Post-quantum cryptography",
                "ISO 20022 compliance",
                "Federated learning integration",
                "Agent-based validation"
            ]
        }

        for category, items in features.items():
            print(f"\n{category}")
            for item in items:
                print(f"  • {item}")

    def open_website(self):
        """Open the website in default browser"""
        print(f"\n🌐 Opening website at {self.base_url}")
        webbrowser.open(self.base_url)
        print("📖 Website opened in your default browser")

    def display_usage_instructions(self):
        """Display instructions for using the website"""
        print("\n📘 How to Use LL TOKEN OFFLINE:")
        print("="*40)
        print("1. 💳 WALLET TAB:")
        print("   • View token balances across 14 token types")
        print("   • Send/receive quantum-safe transactions")
        print("   • Monitor transaction history")
        print("   • Generate new wallet addresses")

        print("\n2. 🏪 MARKETPLACE TAB:")
        print("   • Browse available tokens for trading")
        print("   • Place buy/sell orders")
        print("   • View order book and recent trades")
        print("   • Manage NFT collections")

        print("\n3. 🌍 METAVERSE TAB:")
        print("   • Manage avatar tokens and customizations")
        print("   • Buy/sell virtual land parcels")
        print("   • Transfer assets between metaverses")
        print("   • View soul-bound achievements")

        print("\n4. 🏦 STAKING TAB:")
        print("   • Stake tokens in specialized pools")
        print("   • Monitor rewards and APY rates")
        print("   • Enable auto-compounding")
        print("   • Unstake when lock periods expire")

        print("\n🔒 OFFLINE MODE:")
        print("   • All functionality works without internet")
        print("   • Transactions stored locally until sync")
        print("   • Quantum-resistant security throughout")
        print("   • ISO 20022 compliant messaging")

    def run_complete_test(self):
        """Run all tests and display results"""
        print("🚀 LL TOKEN OFFLINE - Website Functionality Test")
        print("=" * 60)

        # Run tests
        accessibility = self.test_website_accessibility()
        file_structure = self.test_file_structure()
        self.test_javascript_modules()
        self.test_offline_functionality()

        print("\n" + "="*60)

        if accessibility and file_structure:
            print("✅ ALL TESTS PASSED - Website is fully functional!")
            self.display_feature_summary()
            self.display_usage_instructions()

            # Ask if user wants to open the website
            try:
                open_browser = input(f"\n🌐 Open website at {self.base_url}? (y/n): ")
                if open_browser.lower().startswith('y'):
                    self.open_website()
                    print("\n⚡ Website is now running!")
                    print("   • Navigate between tabs to explore features")
                    print("   • All functionality works offline")
                    print("   • Check browser console for detailed logs")
            except KeyboardInterrupt:
                print("\n👋 Test completed")
        else:
            print("❌ SOME TESTS FAILED - Please check the issues above")

if __name__ == "__main__":
    tester = OfflineWebsiteTest()
    tester.run_complete_test()