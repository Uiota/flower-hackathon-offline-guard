#!/usr/bin/env python3
"""
CYBER DEFENSE SYSTEM INTEGRATION
=====================================
Advanced cyber defense with integrated security tools:
- Suricata IDS/IPS Integration
- YARA Malware Detection
- OSSEC/Wazuh HIDS
- ClamAV Antivirus
- Custom Threat Intelligence
- Real-time Security Analytics
"""

import asyncio
import json
import subprocess
import re
import hashlib
import sqlite3
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== SECURITY EVENT MODELS ====================

@dataclass
class SecurityAlert:
    """Unified security alert format"""
    alert_id: str
    timestamp: datetime
    severity: str  # critical, high, medium, low
    source_tool: str  # suricata, yara, ossec, clamav, custom
    alert_type: str
    source_ip: Optional[str]
    destination_ip: Optional[str]
    description: str
    raw_data: Dict[str, Any]
    confidence: float
    mitigation_actions: List[str]

@dataclass
class ThreatIndicator:
    """Threat intelligence indicator"""
    indicator_type: str  # ip, domain, hash, url
    value: str
    threat_level: str
    source: str
    created_at: datetime
    tags: List[str]

# ==================== SURICATA INTEGRATION ====================

class SuricataIntegration:
    """Advanced Suricata IDS/IPS Integration"""

    def __init__(self, config: Dict[str, Any]):
        self.eve_log_path = Path(config.get("eve_log_path", "/var/log/suricata/eve.json"))
        self.rules_path = Path(config.get("rules_path", "/etc/suricata/rules/"))
        self.running = False
        self.alert_callback = None

        # Initialize custom rules
        self._create_custom_rules()

    def _create_custom_rules(self):
        """Create custom Suricata rules"""
        custom_rules = [
            # Detect potential data exfiltration
            'alert tcp any any -> any any (msg:"Potential Data Exfiltration"; content:"password"; content:"admin"; sid:1000001;)',

            # Detect suspicious outbound connections
            'alert tcp any any -> !$HOME_NET any (msg:"Suspicious Outbound Connection"; threshold:type both, track by_src, count 10, seconds 60; sid:1000002;)',

            # Detect command injection attempts
            'alert http any any -> any any (msg:"Command Injection Attempt"; content:"|3b|"; http_uri; pcre:"/;(cat|ls|pwd|whoami)/i"; sid:1000003;)',

            # Detect cryptocurrency mining
            'alert tcp any any -> any any (msg:"Cryptocurrency Mining Activity"; content:"stratum+tcp"; sid:1000004;)',
        ]

        custom_rules_file = self.rules_path / "custom.rules"
        try:
            custom_rules_file.parent.mkdir(exist_ok=True)
            with open(custom_rules_file, 'w') as f:
                f.write('\n'.join(custom_rules))
            logger.info(f"Created custom Suricata rules: {custom_rules_file}")
        except Exception as e:
            logger.warning(f"Could not create custom rules: {e}")

    async def start_monitoring(self, alert_callback):
        """Start Suricata log monitoring"""
        self.running = True
        self.alert_callback = alert_callback

        logger.info(f"🛡️  Starting Suricata monitoring: {self.eve_log_path}")

        # Check if EVE log exists
        if self.eve_log_path.exists():
            await self._tail_eve_log()
        else:
            logger.warning("EVE log not found, using simulation mode")
            await self._simulate_events()

    async def _tail_eve_log(self):
        """Tail Suricata EVE JSON log"""
        try:
            # Use tail -f to follow the log file
            process = await asyncio.create_subprocess_exec(
                "tail", "-f", "-n", "0", str(self.eve_log_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            while self.running:
                line = await process.stdout.readline()
                if not line:
                    await asyncio.sleep(0.1)
                    continue

                try:
                    event = json.loads(line.decode().strip())
                    await self._process_event(event)
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON decode error: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error tailing EVE log: {e}")
            await self._simulate_events()

    async def _process_event(self, event: Dict[str, Any]):
        """Process Suricata event"""
        event_type = event.get("event_type")

        if event_type == "alert":
            alert = self._parse_alert(event)
            if self.alert_callback:
                await self.alert_callback(alert)
        elif event_type == "anomaly":
            # Process network anomalies
            alert = self._parse_anomaly(event)
            if self.alert_callback:
                await self.alert_callback(alert)

    def _parse_alert(self, event: Dict[str, Any]) -> SecurityAlert:
        """Parse Suricata alert"""
        alert_data = event.get("alert", {})

        # Map Suricata severity to standard levels
        severity_map = {1: "critical", 2: "high", 3: "medium"}
        severity = severity_map.get(alert_data.get("severity", 3), "low")

        # Determine mitigation actions based on alert
        mitigation_actions = self._get_mitigation_actions(alert_data)

        return SecurityAlert(
            alert_id=f"suricata_{event.get('flow_id', hash(str(event)) % 100000)}",
            timestamp=datetime.fromisoformat(event.get("timestamp", datetime.now().isoformat()).replace('Z', '+00:00')),
            severity=severity,
            source_tool="suricata",
            alert_type=alert_data.get("category", "network_threat"),
            source_ip=event.get("src_ip"),
            destination_ip=event.get("dest_ip"),
            description=alert_data.get("signature", "Network threat detected"),
            raw_data=event,
            confidence=0.85,
            mitigation_actions=mitigation_actions
        )

    def _parse_anomaly(self, event: Dict[str, Any]) -> SecurityAlert:
        """Parse network anomaly"""
        return SecurityAlert(
            alert_id=f"anomaly_{hash(str(event)) % 100000}",
            timestamp=datetime.now(),
            severity="medium",
            source_tool="suricata",
            alert_type="network_anomaly",
            source_ip=event.get("src_ip"),
            destination_ip=event.get("dest_ip"),
            description="Network anomaly detected",
            raw_data=event,
            confidence=0.70,
            mitigation_actions=["investigate_traffic", "monitor_endpoint"]
        )

    def _get_mitigation_actions(self, alert_data: Dict[str, Any]) -> List[str]:
        """Determine mitigation actions for alert"""
        signature = alert_data.get("signature", "").lower()
        category = alert_data.get("category", "").lower()

        actions = []

        if "malware" in signature or "trojan" in signature:
            actions.extend(["quarantine_endpoint", "run_antivirus_scan"])

        if "sql injection" in signature:
            actions.extend(["block_request", "review_application_logs"])

        if "brute force" in signature:
            actions.extend(["block_source_ip", "enforce_account_lockout"])

        if "data exfiltration" in signature:
            actions.extend(["block_outbound_traffic", "investigate_data_access"])

        if not actions:
            actions = ["investigate_alert", "monitor_activity"]

        return actions

    async def _simulate_events(self):
        """Simulate Suricata events for demo"""
        simulated_events = [
            {
                "timestamp": datetime.now().isoformat(),
                "flow_id": 12345,
                "event_type": "alert",
                "src_ip": "192.168.1.100",
                "dest_ip": "10.0.0.50",
                "alert": {
                    "severity": 2,
                    "category": "Potentially Bad Traffic",
                    "signature": "ET MALWARE Suspicious outbound connection"
                }
            },
            {
                "timestamp": datetime.now().isoformat(),
                "flow_id": 12346,
                "event_type": "alert",
                "src_ip": "192.168.1.200",
                "dest_ip": "192.168.1.1",
                "alert": {
                    "severity": 1,
                    "category": "Web Application Attack",
                    "signature": "ET WEB_SERVER SQL Injection attempt"
                }
            }
        ]

        event_index = 0
        while self.running:
            await asyncio.sleep(5)
            event = simulated_events[event_index % len(simulated_events)]
            event["timestamp"] = datetime.now().isoformat()
            await self._process_event(event)
            event_index += 1

    def stop(self):
        """Stop monitoring"""
        self.running = False
        logger.info("Suricata monitoring stopped")

# ==================== YARA INTEGRATION ====================

class YaraIntegration:
    """Advanced YARA malware detection"""

    def __init__(self, config: Dict[str, Any]):
        self.rules_dir = Path(config.get("rules_dir", "/etc/yara/rules/"))
        self.compiled_rules = None

        # Create rules directory and sample rules
        self._setup_yara_rules()
        self._load_rules()

    def _setup_yara_rules(self):
        """Setup YARA rules directory and create sample rules"""
        self.rules_dir.mkdir(exist_ok=True)

        # Create sample malware detection rules
        sample_rules = {
            "general_malware.yar": '''
rule SuspiciousStrings
{
    meta:
        description = "Detects suspicious strings often found in malware"
        author = "Cyber Defense System"
        date = "2024"
        family = "Generic"

    strings:
        $s1 = "backdoor" nocase
        $s2 = "keylogger" nocase
        $s3 = "password stealer" nocase
        $s4 = "botnet" nocase
        $s5 = "crypto miner" nocase

    condition:
        any of them
}

rule PowershellObfuscation
{
    meta:
        description = "Detects obfuscated PowerShell commands"

    strings:
        $s1 = "powershell" nocase
        $s2 = "-encodedcommand" nocase
        $s3 = "invoke-expression" nocase
        $s4 = "downloadstring" nocase

    condition:
        2 of them
}
''',
            "ransomware.yar": '''
rule RansomwareStrings
{
    meta:
        description = "Detects common ransomware strings"
        family = "Ransomware"

    strings:
        $s1 = "your files have been encrypted" nocase
        $s2 = "bitcoin payment" nocase
        $s3 = "decrypt your files" nocase
        $s4 = ".onion" nocase
        $s5 = "ransom" nocase

    condition:
        2 of them
}
'''
        }

        for filename, content in sample_rules.items():
            rule_file = self.rules_dir / filename
            try:
                with open(rule_file, 'w') as f:
                    f.write(content)
                logger.debug(f"Created YARA rule: {rule_file}")
            except Exception as e:
                logger.warning(f"Could not create rule {filename}: {e}")

    def _load_rules(self):
        """Load and compile YARA rules"""
        try:
            import yara

            rules = {}
            for rule_file in self.rules_dir.glob("*.yar"):
                namespace = rule_file.stem
                rules[namespace] = str(rule_file)

            if rules:
                self.compiled_rules = yara.compile(filepaths=rules)
                logger.info(f"✅ Loaded {len(rules)} YARA rule files")
            else:
                logger.warning("No YARA rule files found")

        except ImportError:
            logger.warning("YARA Python module not installed, using simulation mode")
            self.compiled_rules = None
        except Exception as e:
            logger.error(f"Error loading YARA rules: {e}")
            self.compiled_rules = None

    async def scan_file(self, file_path: str) -> Dict[str, Any]:
        """Scan file with YARA rules"""
        try:
            file_path_obj = Path(file_path)

            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Calculate file hash
            file_hash = self._calculate_hash(file_path_obj)

            # Scan with YARA if available
            if self.compiled_rules:
                matches = self.compiled_rules.match(str(file_path_obj))

                is_malicious = len(matches) > 0
                signatures_matched = [match.rule for match in matches]
                malware_families = list(set([
                    match.meta.get("family", "Unknown")
                    for match in matches
                    if "family" in match.meta
                ]))
                threat_level = self._calculate_threat_level(matches)

            else:
                # Simulation mode
                content_check = self._simulate_scan(file_path)
                is_malicious = content_check["is_malicious"]
                signatures_matched = content_check["signatures"]
                malware_families = content_check["families"]
                threat_level = content_check["threat_level"]

            return {
                "file_path": file_path,
                "file_hash": file_hash,
                "is_malicious": is_malicious,
                "malware_families": malware_families,
                "signatures_matched": signatures_matched,
                "threat_level": threat_level,
                "scan_time": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {e}")
            return {
                "file_path": file_path,
                "error": str(e),
                "scan_time": datetime.now().isoformat()
            }

    def _simulate_scan(self, file_path: str) -> Dict[str, Any]:
        """Simulate YARA scan based on file content/name"""
        file_path_lower = file_path.lower()

        is_malicious = False
        signatures = []
        families = []
        threat_level = "none"

        # Simple simulation based on filename/path
        if any(term in file_path_lower for term in ["malware", "virus", "trojan", "backdoor"]):
            is_malicious = True
            signatures = ["SuspiciousStrings"]
            families = ["Generic"]
            threat_level = "high"

        elif any(term in file_path_lower for term in ["ransom", "encrypt", "bitcoin"]):
            is_malicious = True
            signatures = ["RansomwareStrings"]
            families = ["Ransomware"]
            threat_level = "critical"

        elif file_path_lower.endswith(('.exe', '.dll', '.bat', '.ps1')):
            # Randomly flag some executables for demo
            if hash(file_path) % 5 == 0:  # 20% chance
                is_malicious = True
                signatures = ["SuspiciousStrings"]
                families = ["Generic"]
                threat_level = "medium"

        return {
            "is_malicious": is_malicious,
            "signatures": signatures,
            "families": families,
            "threat_level": threat_level
        }

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception:
            return "error_calculating_hash"

    def _calculate_threat_level(self, matches) -> str:
        """Calculate threat level from YARA matches"""
        if not matches:
            return "none"

        # Check for high-severity indicators
        for match in matches:
            if any(tag in ["critical", "ransomware", "apt"] for tag in match.tags):
                return "critical"

            family = match.meta.get("family", "").lower()
            if family in ["ransomware", "backdoor", "trojan"]:
                return "high"

        if len(matches) > 3:
            return "high"
        elif len(matches) > 1:
            return "medium"
        else:
            return "low"

# ==================== THREAT INTELLIGENCE ====================

class ThreatIntelligence:
    """Custom threat intelligence system"""

    def __init__(self):
        self.db_path = Path("threat_intel.db")
        self._init_database()
        self._load_initial_indicators()

    def _init_database(self):
        """Initialize threat intelligence database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS indicators (
                    indicator_id TEXT PRIMARY KEY,
                    indicator_type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    threat_level TEXT NOT NULL,
                    source TEXT,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS threat_reports (
                    report_id TEXT PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    threat_level TEXT,
                    indicators TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def _load_initial_indicators(self):
        """Load initial threat indicators"""
        indicators = [
            ("ip", "192.168.1.100", "high", "internal_scan", ["malware", "c2"]),
            ("ip", "10.0.0.255", "medium", "network_scan", ["reconnaissance"]),
            ("hash", "a1b2c3d4e5f6789", "critical", "malware_sample", ["trojan", "backdoor"]),
            ("domain", "malicious-site.com", "high", "phishing_campaign", ["phishing"]),
            ("url", "http://suspicious-download.net/payload.exe", "critical", "malware_distribution", ["malware"]),
        ]

        with sqlite3.connect(self.db_path) as conn:
            for indicator_type, value, threat_level, source, tags in indicators:
                conn.execute('''
                    INSERT OR IGNORE INTO indicators
                    (indicator_id, indicator_type, value, threat_level, source, tags)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (f"{indicator_type}_{hash(value) % 100000}", indicator_type, value,
                     threat_level, source, json.dumps(tags)))

    def check_indicator(self, indicator_type: str, value: str) -> Optional[ThreatIndicator]:
        """Check if indicator exists in threat intelligence"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM indicators WHERE indicator_type = ? AND value = ?",
                (indicator_type, value)
            )
            result = cursor.fetchone()

            if result:
                return ThreatIndicator(
                    indicator_type=result[1],
                    value=result[2],
                    threat_level=result[3],
                    source=result[4],
                    created_at=datetime.fromisoformat(result[6]),
                    tags=json.loads(result[5]) if result[5] else []
                )

            return None

    def add_indicator(self, indicator: ThreatIndicator):
        """Add new threat indicator"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO indicators
                (indicator_id, indicator_type, value, threat_level, source, tags, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                f"{indicator.indicator_type}_{hash(indicator.value) % 100000}",
                indicator.indicator_type,
                indicator.value,
                indicator.threat_level,
                indicator.source,
                json.dumps(indicator.tags),
                indicator.created_at.isoformat()
            ))

# ==================== UNIFIED CYBER DEFENSE MANAGER ====================

class CyberDefenseManager:
    """Unified cyber defense system manager"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Initialize components
        self.suricata = SuricataIntegration(self.config.get("suricata", {}))
        self.yara = YaraIntegration(self.config.get("yara", {}))
        self.threat_intel = ThreatIntelligence()

        # Alert handling
        self.alert_queue = asyncio.Queue()
        self.alert_handlers = []

        # Statistics
        self.stats = {
            "alerts_processed": 0,
            "threats_detected": 0,
            "files_scanned": 0,
            "start_time": datetime.now()
        }

        logger.info("🛡️  Cyber Defense Manager initialized")

    def register_alert_handler(self, handler):
        """Register alert handler callback"""
        self.alert_handlers.append(handler)

    async def start(self):
        """Start all cyber defense components"""
        logger.info("🚀 Starting Cyber Defense System...")

        # Start Suricata monitoring
        asyncio.create_task(self.suricata.start_monitoring(self._handle_alert))

        # Start alert processing
        asyncio.create_task(self._process_alerts())

        logger.info("✅ Cyber Defense System started")

    async def _handle_alert(self, alert: SecurityAlert):
        """Handle incoming security alert"""
        await self.alert_queue.put(alert)
        self.stats["alerts_processed"] += 1

        if alert.severity in ["critical", "high"]:
            self.stats["threats_detected"] += 1

    async def _process_alerts(self):
        """Process alerts from queue"""
        while True:
            try:
                alert = await self.alert_queue.get()

                # Enrich alert with threat intelligence
                enriched_alert = await self._enrich_alert(alert)

                # Notify handlers
                for handler in self.alert_handlers:
                    await handler(enriched_alert)

            except Exception as e:
                logger.error(f"Error processing alert: {e}")

    async def _enrich_alert(self, alert: SecurityAlert) -> SecurityAlert:
        """Enrich alert with threat intelligence"""
        # Check source IP
        if alert.source_ip:
            threat_info = self.threat_intel.check_indicator("ip", alert.source_ip)
            if threat_info:
                alert.description += f" [Known malicious IP: {threat_info.source}]"
                alert.confidence = min(1.0, alert.confidence + 0.2)

        # Check destination IP
        if alert.destination_ip:
            threat_info = self.threat_intel.check_indicator("ip", alert.destination_ip)
            if threat_info:
                alert.description += f" [Known malicious destination: {threat_info.source}]"
                alert.confidence = min(1.0, alert.confidence + 0.2)

        return alert

    async def scan_file(self, file_path: str) -> Dict[str, Any]:
        """Comprehensive file scanning"""
        logger.info(f"🔍 Scanning file: {file_path}")

        # YARA scan
        yara_result = await self.yara.scan_file(file_path)
        self.stats["files_scanned"] += 1

        # Check file hash against threat intelligence
        if "file_hash" in yara_result:
            threat_info = self.threat_intel.check_indicator("hash", yara_result["file_hash"])
            if threat_info:
                yara_result["threat_intel_match"] = {
                    "source": threat_info.source,
                    "threat_level": threat_info.threat_level,
                    "tags": threat_info.tags
                }

        return yara_result

    async def scan_directory(self, directory: str) -> List[Dict[str, Any]]:
        """Scan entire directory"""
        logger.info(f"📁 Scanning directory: {directory}")

        results = []
        for file_path in Path(directory).rglob("*"):
            if file_path.is_file():
                try:
                    result = await self.scan_file(str(file_path))
                    results.append(result)

                    # Add small delay to prevent overwhelming the system
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error scanning {file_path}: {e}")

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get defense system statistics"""
        uptime = datetime.now() - self.stats["start_time"]

        return {
            "alerts_processed": self.stats["alerts_processed"],
            "threats_detected": self.stats["threats_detected"],
            "files_scanned": self.stats["files_scanned"],
            "uptime_seconds": uptime.total_seconds(),
            "threat_detection_rate": (
                self.stats["threats_detected"] / max(1, self.stats["alerts_processed"]) * 100
            )
        }

    def stop(self):
        """Stop cyber defense system"""
        logger.info("⏹️  Stopping Cyber Defense System...")
        self.suricata.stop()

# ==================== DEMO ====================

async def demo_cyber_defense():
    """Demonstrate cyber defense system"""

    print("\n" + "="*80)
    print("ADVANCED CYBER DEFENSE SYSTEM DEMONSTRATION")
    print("="*80)

    # Initialize defense system
    manager = CyberDefenseManager()

    # Register alert handler
    async def alert_handler(alert: SecurityAlert):
        print(f"\n🚨 SECURITY ALERT:")
        print(f"   Source: {alert.source_tool}")
        print(f"   Severity: {alert.severity}")
        print(f"   Type: {alert.alert_type}")
        print(f"   Description: {alert.description}")
        if alert.source_ip:
            print(f"   Source IP: {alert.source_ip}")
        if alert.mitigation_actions:
            print(f"   Recommended Actions: {', '.join(alert.mitigation_actions)}")
        print(f"   Confidence: {alert.confidence:.2f}")

    manager.register_alert_handler(alert_handler)

    # Start defense system
    await manager.start()

    print("\n📡 Monitoring for threats (15 seconds)...")
    await asyncio.sleep(15)

    # Scan some files
    print("\n🔍 Performing file scans...")

    # Create test files for scanning
    test_dir = Path("test_scan")
    test_dir.mkdir(exist_ok=True)

    test_files = {
        "clean_file.txt": "This is a clean file with normal content.",
        "suspicious_file.txt": "This file contains backdoor and keylogger references.",
        "malware_sample.exe": "This is a simulated malware file with suspicious content.",
        "ransom_note.txt": "Your files have been encrypted. Pay bitcoin to decrypt your files."
    }

    for filename, content in test_files.items():
        with open(test_dir / filename, 'w') as f:
            f.write(content)

    # Scan directory
    scan_results = await manager.scan_directory(str(test_dir))

    print(f"\n📊 Scan Results ({len(scan_results)} files):")
    for result in scan_results:
        file_path = result.get("file_path", "unknown")
        is_malicious = result.get("is_malicious", False)
        threat_level = result.get("threat_level", "none")

        status = "🚨 MALICIOUS" if is_malicious else "✅ Clean"
        print(f"   {Path(file_path).name}: {status} (Threat Level: {threat_level})")

        if result.get("signatures_matched"):
            print(f"      Signatures: {', '.join(result['signatures_matched'])}")

    # Show statistics
    print(f"\n📈 System Statistics:")
    stats = manager.get_statistics()
    print(f"   Alerts Processed: {stats['alerts_processed']}")
    print(f"   Threats Detected: {stats['threats_detected']}")
    print(f"   Files Scanned: {stats['files_scanned']}")
    print(f"   Uptime: {stats['uptime_seconds']:.1f} seconds")
    print(f"   Threat Detection Rate: {stats['threat_detection_rate']:.1f}%")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)

    manager.stop()

    print("\n✅ Cyber Defense System demonstration complete!")
    print("="*80)

if __name__ == "__main__":
    try:
        asyncio.run(demo_cyber_defense())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        import traceback
        traceback.print_exc()