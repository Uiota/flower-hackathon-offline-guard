#!/usr/bin/env python3
"""
Memory Guardian Agent System
Development Agent: Monitors system health, updates, and maintains the platform
Research Agent: Analyzes cognitive data, detects patterns, and provides insights
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import sqlite3
import hashlib


@dataclass
class AgentTask:
    """Task for an agent to execute"""
    task_id: str
    agent_type: str  # 'dev' or 'research'
    priority: int  # 1-5, 5 highest
    task_type: str
    description: str
    status: str  # 'pending', 'in_progress', 'completed', 'failed'
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict] = None
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class DevelopmentAgent:
    """
    Development Agent - System Maintenance & Updates

    Responsibilities:
    - Monitor system health and performance
    - Check for security updates
    - Maintain data integrity
    - Optimize database performance
    - Generate system reports
    - Handle backup operations
    """

    def __init__(self, db_path: str = "memory_guardian.db"):
        self.agent_id = "dev_agent_001"
        self.db_path = db_path
        self.tasks = []
        self.last_health_check = None

        print(f"✅ Development Agent {self.agent_id} initialized")

    def system_health_check(self) -> Dict:
        """
        Perform comprehensive system health check
        """
        print(f"🔧 [{self.agent_id}] Running system health check...")

        health_report = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "checks": []
        }

        # Check 1: Database integrity
        db_check = self._check_database_integrity()
        health_report["checks"].append(db_check)

        # Check 2: File system
        fs_check = self._check_file_system()
        health_report["checks"].append(fs_check)

        # Check 3: Encryption system
        encryption_check = self._check_encryption_health()
        health_report["checks"].append(encryption_check)

        # Check 4: Memory usage
        memory_check = self._check_memory_usage()
        health_report["checks"].append(memory_check)

        # Overall health status
        all_passed = all(check["status"] == "healthy" for check in health_report["checks"])
        health_report["overall_status"] = "healthy" if all_passed else "needs_attention"

        self.last_health_check = datetime.now()

        return health_report

    def _check_database_integrity(self) -> Dict:
        """Check database integrity"""
        try:
            if not os.path.exists(self.db_path):
                return {
                    "check": "database_integrity",
                    "status": "warning",
                    "message": "Database file not found",
                    "recommendation": "Initialize database"
                }

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Run integrity check
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()

            conn.close()

            if result and result[0] == "ok":
                return {
                    "check": "database_integrity",
                    "status": "healthy",
                    "message": "Database integrity verified"
                }
            else:
                return {
                    "check": "database_integrity",
                    "status": "critical",
                    "message": "Database integrity issues detected",
                    "recommendation": "Restore from backup"
                }

        except Exception as e:
            return {
                "check": "database_integrity",
                "status": "error",
                "message": str(e)
            }

    def _check_file_system(self) -> Dict:
        """Check file system health"""
        try:
            # Check if we can write to the directory
            test_file = ".system_health_test"
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)

            return {
                "check": "file_system",
                "status": "healthy",
                "message": "File system read/write operational"
            }
        except Exception as e:
            return {
                "check": "file_system",
                "status": "critical",
                "message": f"File system issue: {str(e)}"
            }

    def _check_encryption_health(self) -> Dict:
        """Verify encryption system is working"""
        try:
            from cryptography.fernet import Fernet

            # Test encryption/decryption
            key = Fernet.generate_key()
            cipher = Fernet(key)
            test_data = b"Memory Guardian encryption test"

            encrypted = cipher.encrypt(test_data)
            decrypted = cipher.decrypt(encrypted)

            if decrypted == test_data:
                return {
                    "check": "encryption_system",
                    "status": "healthy",
                    "message": "Encryption system operational"
                }
            else:
                return {
                    "check": "encryption_system",
                    "status": "critical",
                    "message": "Encryption/decryption mismatch"
                }

        except Exception as e:
            return {
                "check": "encryption_system",
                "status": "error",
                "message": f"Encryption test failed: {str(e)}"
            }

    def _check_memory_usage(self) -> Dict:
        """Check application memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb < 100:
                status = "healthy"
            elif memory_mb < 500:
                status = "warning"
            else:
                status = "critical"

            return {
                "check": "memory_usage",
                "status": status,
                "message": f"Using {memory_mb:.1f} MB",
                "recommendation": "Optimize if exceeds 500 MB"
            }
        except ImportError:
            return {
                "check": "memory_usage",
                "status": "unknown",
                "message": "psutil not available"
            }

    def optimize_database(self) -> Dict:
        """
        Optimize database performance
        """
        print(f"🔧 [{self.agent_id}] Optimizing database...")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Vacuum database
            cursor.execute("VACUUM")

            # Analyze tables for query optimization
            cursor.execute("ANALYZE")

            conn.commit()
            conn.close()

            return {
                "status": "success",
                "message": "Database optimized successfully",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def create_backup(self, backup_dir: str = "./backups") -> Dict:
        """
        Create database backup
        """
        print(f"🔧 [{self.agent_id}] Creating backup...")

        try:
            os.makedirs(backup_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"memory_guardian_backup_{timestamp}.db")

            # Copy database
            import shutil
            shutil.copy2(self.db_path, backup_path)

            # Calculate checksum
            with open(backup_path, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()

            return {
                "status": "success",
                "backup_path": backup_path,
                "checksum": checksum,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def generate_system_report(self) -> Dict:
        """
        Generate comprehensive system report
        """
        print(f"🔧 [{self.agent_id}] Generating system report...")

        report = {
            "report_id": hashlib.sha256(
                f"report_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16],
            "generated_at": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "health_check": self.system_health_check(),
            "database_stats": self._get_database_stats(),
            "recommendations": []
        }

        # Add recommendations based on health check
        if report["health_check"]["overall_status"] != "healthy":
            for check in report["health_check"]["checks"]:
                if check["status"] != "healthy" and "recommendation" in check:
                    report["recommendations"].append({
                        "priority": "high" if check["status"] == "critical" else "medium",
                        "issue": check["check"],
                        "recommendation": check["recommendation"]
                    })

        return report

    def _get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            stats = {}

            # Count records in each table
            tables = ["cognitive_assessments", "property_records", "trusted_contacts", "alert_logs"]
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]

            # Database size
            stats["database_size_mb"] = os.path.getsize(self.db_path) / 1024 / 1024

            conn.close()

            return stats

        except Exception as e:
            return {"error": str(e)}

    def auto_maintenance(self) -> Dict:
        """
        Perform automated maintenance tasks
        """
        print(f"🔧 [{self.agent_id}] Running auto-maintenance...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "tasks": []
        }

        # Task 1: Health check
        health = self.system_health_check()
        results["tasks"].append({
            "task": "health_check",
            "result": health["overall_status"]
        })

        # Task 2: Optimize database
        optimization = self.optimize_database()
        results["tasks"].append({
            "task": "database_optimization",
            "result": optimization["status"]
        })

        # Task 3: Create backup (weekly)
        if self._should_backup():
            backup = self.create_backup()
            results["tasks"].append({
                "task": "backup",
                "result": backup["status"]
            })

        return results

    def _should_backup(self) -> bool:
        """Determine if backup is needed"""
        backup_dir = "./backups"
        if not os.path.exists(backup_dir):
            return True

        # Check if last backup was more than 7 days ago
        backups = [f for f in os.listdir(backup_dir) if f.startswith("memory_guardian_backup")]
        if not backups:
            return True

        latest_backup = max(backups)
        backup_time_str = latest_backup.split('_')[-2] + latest_backup.split('_')[-1].replace('.db', '')

        try:
            backup_time = datetime.strptime(backup_time_str, "%Y%m%d%H%M%S")
            return (datetime.now() - backup_time).days >= 7
        except:
            return True


class ResearchAgent:
    """
    Research Agent - Cognitive Data Analysis & Insights

    Responsibilities:
    - Analyze cognitive assessment patterns
    - Detect early signs of cognitive decline
    - Generate personalized insights
    - Recommend exercise adjustments
    - Contribute to FL research
    - Generate health reports
    """

    def __init__(self, db_path: str = "memory_guardian.db"):
        self.agent_id = "research_agent_001"
        self.db_path = db_path
        self.analysis_cache = {}

        print(f"✅ Research Agent {self.agent_id} initialized")

    def analyze_cognitive_trends(self, user_id: str, days: int = 90) -> Dict:
        """
        Analyze cognitive health trends over time
        """
        print(f"🔬 [{self.agent_id}] Analyzing cognitive trends for user {user_id}...")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            cursor.execute("""
                SELECT timestamp, memory_score, reaction_time_ms,
                       pattern_recognition_score, problem_solving_score,
                       overall_score, baseline_deviation
                FROM cognitive_assessments
                WHERE user_id = ? AND timestamp > ?
                ORDER BY timestamp ASC
            """, (user_id, cutoff_date))

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return {
                    "status": "insufficient_data",
                    "message": "Need at least 3 assessments for trend analysis"
                }

            # Parse data
            timestamps = []
            memory_scores = []
            reaction_times = []
            pattern_scores = []
            problem_scores = []
            overall_scores = []

            for row in rows:
                timestamps.append(row[0])
                memory_scores.append(row[1])
                reaction_times.append(row[2])
                pattern_scores.append(row[3])
                problem_scores.append(row[4])
                overall_scores.append(row[5])

            # Calculate trends
            trends = {
                "memory": self._calculate_trend(memory_scores),
                "reaction_time": self._calculate_trend(reaction_times, lower_is_better=True),
                "pattern_recognition": self._calculate_trend(pattern_scores),
                "problem_solving": self._calculate_trend(problem_scores),
                "overall": self._calculate_trend(overall_scores)
            }

            # Detect anomalies
            anomalies = self._detect_anomalies(overall_scores)

            # Generate insights
            insights = self._generate_insights(trends, anomalies)

            return {
                "user_id": user_id,
                "analysis_period_days": days,
                "total_assessments": len(rows),
                "trends": trends,
                "anomalies": anomalies,
                "insights": insights,
                "risk_level": self._assess_risk_level(trends),
                "recommendations": self._generate_recommendations(trends),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def _calculate_trend(self, values: List[float], lower_is_better: bool = False) -> Dict:
        """Calculate trend direction and strength"""
        if len(values) < 3:
            return {"direction": "unknown", "strength": 0}

        # Simple linear regression
        n = len(values)
        x = list(range(n))
        y = values

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator

        # Adjust for lower_is_better metrics
        if lower_is_better:
            slope = -slope

        # Determine direction and strength
        if abs(slope) < 0.1:
            direction = "stable"
            strength = abs(slope) * 10
        elif slope > 0:
            direction = "improving"
            strength = min(slope * 10, 100)
        else:
            direction = "declining"
            strength = min(abs(slope) * 10, 100)

        return {
            "direction": direction,
            "strength": round(strength, 2),
            "slope": round(slope, 4),
            "recent_average": round(sum(values[-7:]) / min(7, len(values)), 2),
            "overall_average": round(y_mean, 2)
        }

    def _detect_anomalies(self, scores: List[float]) -> List[Dict]:
        """Detect unusual drops in scores"""
        anomalies = []

        if len(scores) < 5:
            return anomalies

        # Calculate mean and std dev
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        std_dev = variance ** 0.5

        # Detect scores more than 2 std devs below mean
        for i, score in enumerate(scores):
            if score < (mean - 2 * std_dev):
                anomalies.append({
                    "index": i,
                    "score": score,
                    "deviation": round((score - mean) / std_dev, 2),
                    "severity": "high" if score < (mean - 3 * std_dev) else "medium"
                })

        return anomalies

    def _generate_insights(self, trends: Dict, anomalies: List[Dict]) -> List[str]:
        """Generate human-readable insights"""
        insights = []

        # Trend insights
        for category, trend in trends.items():
            if category == "overall":
                continue

            if trend["direction"] == "declining" and trend["strength"] > 30:
                insights.append(
                    f"⚠️ {category.replace('_', ' ').title()} shows declining trend "
                    f"(strength: {trend['strength']:.1f}%)"
                )
            elif trend["direction"] == "improving" and trend["strength"] > 30:
                insights.append(
                    f"✅ {category.replace('_', ' ').title()} shows improvement "
                    f"(strength: {trend['strength']:.1f}%)"
                )

        # Anomaly insights
        if anomalies:
            high_severity = [a for a in anomalies if a["severity"] == "high"]
            if high_severity:
                insights.append(
                    f"🚨 {len(high_severity)} significant performance drop(s) detected"
                )

        # Overall assessment
        overall_trend = trends.get("overall", {})
        if overall_trend.get("direction") == "declining":
            insights.append(
                "📉 Overall cognitive performance trending downward - recommend healthcare consultation"
            )
        elif overall_trend.get("direction") == "stable":
            insights.append(
                "📊 Cognitive performance is stable - continue regular exercises"
            )
        elif overall_trend.get("direction") == "improving":
            insights.append(
                "📈 Cognitive performance improving - excellent progress!"
            )

        return insights

    def _assess_risk_level(self, trends: Dict) -> str:
        """Assess overall cognitive risk level"""
        overall_trend = trends.get("overall", {})

        if overall_trend.get("direction") == "declining":
            if overall_trend.get("strength", 0) > 50:
                return "high"
            elif overall_trend.get("strength", 0) > 25:
                return "medium"
            else:
                return "low"
        else:
            return "none"

    def _generate_recommendations(self, trends: Dict) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []

        # Specific category recommendations
        for category, trend in trends.items():
            if category == "overall":
                continue

            if trend["direction"] == "declining":
                if category == "memory":
                    recommendations.append(
                        "Increase memory exercises - try word association and sequence memorization"
                    )
                elif category == "pattern_recognition":
                    recommendations.append(
                        "Practice pattern puzzles and visual recognition exercises"
                    )
                elif category == "problem_solving":
                    recommendations.append(
                        "Engage in logic puzzles and strategic thinking games"
                    )
                elif category == "reaction_time":
                    recommendations.append(
                        "Practice reaction time exercises and consider physical activity"
                    )

        # General recommendations
        overall_trend = trends.get("overall", {})
        if overall_trend.get("direction") == "declining":
            recommendations.extend([
                "Consult with healthcare provider for comprehensive evaluation",
                "Ensure adequate sleep (7-9 hours per night)",
                "Maintain regular physical exercise routine",
                "Consider social engagement activities"
            ])

        return recommendations

    def generate_research_insights(self, user_id: str) -> Dict:
        """
        Generate insights for federated learning research
        """
        print(f"🔬 [{self.agent_id}] Generating research insights...")

        analysis = self.analyze_cognitive_trends(user_id, days=180)

        if analysis.get("status") == "error":
            return analysis

        # Anonymize and prepare for FL contribution
        research_data = {
            "age_group": "60-70",  # Generalized
            "assessment_count": analysis.get("total_assessments", 0),
            "trend_summary": {
                "overall_direction": analysis["trends"]["overall"]["direction"],
                "risk_level": analysis["risk_level"]
            },
            "anomaly_count": len(analysis.get("anomalies", [])),
            "contribution_id": hashlib.sha256(
                f"{user_id}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]
        }

        return {
            "research_contribution": research_data,
            "fl_ready": True,
            "privacy_level": "high",
            "timestamp": datetime.now().isoformat()
        }


class AgentCoordinator:
    """Coordinate between dev and research agents"""

    def __init__(self, db_path: str = "memory_guardian.db"):
        self.dev_agent = DevelopmentAgent(db_path)
        self.research_agent = ResearchAgent(db_path)
        self.task_queue = []

        print("✅ Agent Coordinator initialized")

    def run_daily_maintenance(self, user_id: str) -> Dict:
        """
        Run coordinated daily maintenance and analysis
        """
        print("\n" + "=" * 80)
        print("RUNNING DAILY MAINTENANCE AND ANALYSIS")
        print("=" * 80)

        results = {
            "timestamp": datetime.now().isoformat(),
            "tasks": []
        }

        # Dev agent tasks
        print("\n1️⃣ Development Agent Tasks:")
        dev_results = self.dev_agent.auto_maintenance()
        results["tasks"].append({
            "agent": "dev",
            "results": dev_results
        })

        # Research agent tasks
        print("\n2️⃣ Research Agent Tasks:")
        research_results = self.research_agent.analyze_cognitive_trends(user_id)
        results["tasks"].append({
            "agent": "research",
            "results": research_results
        })

        # Generate combined report
        print("\n3️⃣ Generating Combined Report:")
        system_report = self.dev_agent.generate_system_report()
        results["system_report"] = system_report

        print("\n✅ Daily maintenance complete!")

        return results


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("MEMORY GUARDIAN AGENT SYSTEM")
    print("=" * 80)

    # Initialize coordinator
    coordinator = AgentCoordinator()

    # Run daily maintenance
    results = coordinator.run_daily_maintenance(user_id="user_demo_001")

    # Display results
    print("\n" + "=" * 80)
    print("MAINTENANCE RESULTS")
    print("=" * 80)

    for task in results["tasks"]:
        print(f"\n{task['agent'].upper()} AGENT:")
        print(json.dumps(task['results'], indent=2))

    print("\n✅ Agent system demo complete!")