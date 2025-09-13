#!/usr/bin/env python3
"""
Peer-to-Peer Team Coordination System
Enables decentralized team collaboration without central servers
"""

import hashlib
import json
import time
import socket
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import uuid

@dataclass
class TeamMember:
    id: str
    name: str
    skills: List[str]
    location: str
    guardian_class: str
    public_key: str  # For signing contributions
    last_seen: float
    contributions: List[str]
    availability_status: str  # "available", "busy", "offline"

@dataclass
class CollaborationRequest:
    id: str
    requester_id: str
    project_name: str
    required_skills: List[str]
    description: str
    deadline: str
    timestamp: float
    responses: List[str]

class P2PMeshCoordinator:
    def __init__(self, port: int = 8888):
        self.port = port
        self.node_id = str(uuid.uuid4())
        self.members: Dict[str, TeamMember] = {}
        self.collaboration_requests: Dict[str, CollaborationRequest] = {}
        self.peer_connections = set()
        self.running = False
        
        # Flower AI integration data
        self.flower_hackathon_info = {
            "event_name": "Flower AI Day 2025",
            "date": "2025-09-25",
            "location": "San Francisco",
            "registration_deadline": "2025-08-25",
            "team_size_limit": 5,
            "focus_areas": [
                "federated_learning",
                "privacy_preserving_ai",
                "edge_computing",
                "decentralized_systems"
            ]
        }
        
    def start_node(self):
        """Start the P2P node for mesh coordination"""
        self.running = True
        
        # Start discovery service
        discovery_thread = threading.Thread(target=self._discovery_service)
        discovery_thread.daemon = True
        discovery_thread.start()
        
        # Start message handler
        handler_thread = threading.Thread(target=self._message_handler)
        handler_thread.daemon = True
        handler_thread.start()
        
        print(f"🌐 P2P Team Coordinator started on node {self.node_id[:8]}")
        
    def register_member(self, name: str, skills: List[str], location: str, 
                       guardian_class: str, public_key: str) -> str:
        """Register a new team member in the mesh"""
        member_id = hashlib.sha256(f"{name}{time.time()}".encode()).hexdigest()[:16]
        
        member = TeamMember(
            id=member_id,
            name=name,
            skills=skills,
            location=location,
            guardian_class=guardian_class,
            public_key=public_key,
            last_seen=time.time(),
            contributions=[],
            availability_status="available"
        )
        
        self.members[member_id] = member
        self._broadcast_member_update(member)
        
        print(f"✅ Registered {name} as Guardian {guardian_class}")
        return member_id
    
    def create_collaboration_request(self, requester_id: str, project_name: str,
                                   required_skills: List[str], description: str,
                                   deadline: str) -> str:
        """Create a new collaboration request"""
        request_id = str(uuid.uuid4())
        
        request = CollaborationRequest(
            id=request_id,
            requester_id=requester_id,
            project_name=project_name,
            required_skills=required_skills,
            description=description,
            deadline=deadline,
            timestamp=time.time(),
            responses=[]
        )
        
        self.collaboration_requests[request_id] = request
        self._broadcast_collaboration_request(request)
        
        return request_id
    
    def find_skill_matches(self, required_skills: List[str], 
                          location_preference: Optional[str] = None) -> List[TeamMember]:
        """Find team members matching required skills"""
        matches = []
        
        for member in self.members.values():
            if member.availability_status == "offline":
                continue
                
            skill_overlap = set(required_skills) & set(member.skills)
            if skill_overlap:
                score = len(skill_overlap) / len(required_skills)
                
                # Bonus for location match
                if location_preference and member.location.lower() == location_preference.lower():
                    score += 0.2
                    
                # Bonus for Flower AI relevant skills
                flower_skills = {"federated_learning", "pytorch", "tensorflow", "privacy", "edge"}
                if any(skill in flower_skills for skill in member.skills):
                    score += 0.1
                
                matches.append((member, score))
        
        # Sort by match score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        return [member for member, _ in matches]
    
    def create_flower_hackathon_team(self, team_lead_id: str) -> Dict:
        """Create a specialized team for Flower AI hackathon"""
        if team_lead_id not in self.members:
            return {"error": "Team lead not found"}
            
        # Required skills for Flower AI hackathon
        flower_skills = [
            "python", "machine_learning", "federated_learning",
            "pytorch", "tensorflow", "docker", "kubernetes"
        ]
        
        # Find best matches
        potential_members = self.find_skill_matches(flower_skills)
        
        # Create balanced team (max 5 members including lead)
        team = [self.members[team_lead_id]]
        skill_coverage = set(self.members[team_lead_id].skills)
        
        for member in potential_members:
            if len(team) >= 5:
                break
            if member.id == team_lead_id:
                continue
                
            # Add member if they bring new skills
            new_skills = set(member.skills) - skill_coverage
            if new_skills:
                team.append(member)
                skill_coverage.update(member.skills)
        
        team_info = {
            "team_id": str(uuid.uuid4()),
            "hackathon": "Flower AI Day 2025",
            "members": [asdict(member) for member in team],
            "total_skills": list(skill_coverage),
            "flower_skill_coverage": len(set(flower_skills) & skill_coverage),
            "created": datetime.now(timezone.utc).isoformat()
        }
        
        return team_info
    
    def coordinate_travel(self, event: str, departure_location: str) -> Dict:
        """Coordinate travel arrangements for hackathon attendance"""
        travel_groups = {}
        
        for member in self.members.values():
            if member.availability_status == "offline":
                continue
                
            # Group by proximity (simplified - in reality would use geolocation)
            region = self._get_region(member.location)
            
            if region not in travel_groups:
                travel_groups[region] = {
                    "members": [],
                    "potential_drivers": [],
                    "hotel_sharing": [],
                    "flight_coordination": []
                }
            
            travel_groups[region]["members"].append({
                "id": member.id,
                "name": member.name,
                "location": member.location
            })
        
        return {
            "event": event,
            "departure_from": departure_location,
            "travel_groups": travel_groups,
            "coordination_tips": [
                "Use encrypted group chats for coordination",
                "Share travel itineraries only with trusted team members", 
                "Consider carbon offset for flights",
                "Book accommodations near the hackathon venue"
            ]
        }
    
    def generate_hackathon_preparation_plan(self, team_id: str) -> Dict:
        """Generate a preparation plan for Flower AI hackathon"""
        plan = {
            "hackathon": "Flower AI Day 2025",
            "team_id": team_id,
            "preparation_phases": {
                "phase_1_foundation": {
                    "duration": "2 weeks",
                    "tasks": [
                        "Install and explore Flower framework",
                        "Complete Flower tutorials",
                        "Set up development environment",
                        "Study federated learning fundamentals"
                    ]
                },
                "phase_2_specialization": {
                    "duration": "2 weeks", 
                    "tasks": [
                        "Choose specific FL use case (healthcare, finance, IoT)",
                        "Design federated architecture",
                        "Implement basic FL prototype",
                        "Test with simulated clients"
                    ]
                },
                "phase_3_integration": {
                    "duration": "1 week",
                    "tasks": [
                        "Integrate with Offline Guard concepts",
                        "Add offline-first capabilities",
                        "Implement Guardian character evolution",
                        "Prepare presentation materials"
                    ]
                }
            },
            "required_skills_development": [
                "Flower framework proficiency",
                "Federated learning theory",
                "Privacy-preserving techniques",
                "Edge computing optimization"
            ],
            "deliverables": [
                "Working FL prototype",
                "Technical documentation",
                "Demo presentation",
                "Guardian NFT integration"
            ]
        }
        
        return plan
    
    def _discovery_service(self):
        """Discover other nodes in the mesh"""
        while self.running:
            try:
                # Simplified discovery - in reality would use mDNS or DHT
                # Broadcast presence on local network
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                
                announcement = {
                    "type": "node_announcement",
                    "node_id": self.node_id,
                    "port": self.port,
                    "capabilities": ["team_coordination", "flower_integration"],
                    "timestamp": time.time()
                }
                
                message = json.dumps(announcement).encode()
                sock.sendto(message, ('<broadcast>', self.port))
                sock.close()
                
                time.sleep(30)  # Announce every 30 seconds
                
            except Exception as e:
                print(f"❌ Discovery error: {e}")
                time.sleep(10)
    
    def _message_handler(self):
        """Handle incoming P2P messages"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(('', self.port))
            sock.settimeout(1.0)
            
            while self.running:
                try:
                    data, addr = sock.recvfrom(4096)
                    message = json.loads(data.decode())
                    self._process_message(message, addr)
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"❌ Message handling error: {e}")
                    
        except Exception as e:
            print(f"❌ Socket error: {e}")
    
    def _process_message(self, message: Dict, addr):
        """Process incoming P2P messages"""
        msg_type = message.get("type")
        
        if msg_type == "node_announcement":
            peer_id = message.get("node_id")
            if peer_id != self.node_id:
                self.peer_connections.add(addr[0])
                print(f"🔗 Discovered peer: {peer_id[:8]} at {addr[0]}")
                
        elif msg_type == "member_update":
            member_data = message.get("member")
            if member_data:
                member = TeamMember(**member_data)
                self.members[member.id] = member
                
        elif msg_type == "collaboration_request":
            request_data = message.get("request")
            if request_data:
                request = CollaborationRequest(**request_data)
                self.collaboration_requests[request.id] = request
                print(f"📋 New collaboration request: {request.project_name}")
    
    def _broadcast_member_update(self, member: TeamMember):
        """Broadcast member update to mesh"""
        message = {
            "type": "member_update",
            "member": asdict(member),
            "timestamp": time.time()
        }
        self._broadcast_message(message)
    
    def _broadcast_collaboration_request(self, request: CollaborationRequest):
        """Broadcast collaboration request to mesh"""
        message = {
            "type": "collaboration_request", 
            "request": asdict(request),
            "timestamp": time.time()
        }
        self._broadcast_message(message)
    
    def _broadcast_message(self, message: Dict):
        """Send message to all known peers"""
        message_data = json.dumps(message).encode()
        
        for peer_ip in self.peer_connections:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.sendto(message_data, (peer_ip, self.port))
                sock.close()
            except Exception as e:
                print(f"❌ Broadcast error to {peer_ip}: {e}")
    
    def _get_region(self, location: str) -> str:
        """Get region from location (simplified)"""
        location = location.lower()
        if any(city in location for city in ["san francisco", "sf", "bay area", "silicon valley"]):
            return "sf_bay_area"
        elif any(city in location for city in ["new york", "nyc", "ny"]):
            return "new_york"
        elif any(city in location for city in ["london", "uk", "britain"]):
            return "london"
        else:
            return "other"

if __name__ == "__main__":
    # Example usage
    coordinator = P2PMeshCoordinator()
    coordinator.start_node()
    
    # Register some test members
    alice_id = coordinator.register_member(
        "Alice", ["python", "federated_learning", "pytorch"], 
        "San Francisco", "AIGuardian", "alice_public_key"
    )
    
    bob_id = coordinator.register_member(
        "Bob", ["javascript", "react", "ui/ux", "design"],
        "New York", "UIGuardian", "bob_public_key" 
    )
    
    # Create Flower hackathon team
    team_info = coordinator.create_flower_hackathon_team(alice_id)
    print(f"🌸 Created Flower hackathon team: {json.dumps(team_info, indent=2)}")
    
    # Coordinate travel
    travel_info = coordinator.coordinate_travel("Flower AI Day 2025", "San Francisco")
    print(f"✈️ Travel coordination: {json.dumps(travel_info, indent=2)}")
    
    try:
        input("Press Enter to stop the coordinator...\n")
    finally:
        coordinator.running = False