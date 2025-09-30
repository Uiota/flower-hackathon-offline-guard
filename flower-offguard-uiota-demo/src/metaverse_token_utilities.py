"""
LL TOKEN OFFLINE - Metaverse Utilities and Token Specifications
Comprehensive token economy for virtual worlds and federated learning ecosystems
"""

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from decimal import Decimal
from enum import Enum
import logging

from .quantum_wallet import QuantumWallet, TokenRail
from .guard import sign_blob, new_keypair

logger = logging.getLogger(__name__)


class TokenType(Enum):
    """LL TOKEN OFFLINE token types and their specific utilities."""

    # Core Utility Tokens
    LLT_COMPUTE = "LLT_COMPUTE"      # Computational resource access
    LLT_DATA = "LLT_DATA"            # Data access and sharing rights
    LLT_GOVERNANCE = "LLT_GOV"       # Voting and governance rights
    LLT_REWARD = "LLT_REWARD"        # FL participation rewards

    # Metaverse Utility Tokens
    LLT_AVATAR = "LLT_AVATAR"        # Avatar customization and abilities
    LLT_LAND = "LLT_LAND"            # Virtual land ownership and development
    LLT_ASSET = "LLT_ASSET"          # Virtual asset creation and trading
    LLT_EXPERIENCE = "LLT_EXP"       # Experience and skill progression

    # Economic Tokens
    LLT_STAKE = "LLT_STAKE"          # Staking and validation rewards
    LLT_LIQUIDITY = "LLT_LP"         # Liquidity provision tokens
    LLT_YIELD = "LLT_YIELD"          # Yield farming tokens
    LLT_INSURANCE = "LLT_INS"        # Insurance and protection protocols

    # Social Tokens
    LLT_REPUTATION = "LLT_REP"       # Reputation and social status
    LLT_COLLABORATION = "LLT_COLLAB" # Collaboration and team rewards
    LLT_EDUCATION = "LLT_EDU"        # Educational content access
    LLT_CREATOR = "LLT_CREATE"       # Content creation rewards


@dataclass
class TokenSpecification:
    """Detailed specification for each LL TOKEN type."""

    token_type: TokenType
    name: str
    symbol: str
    total_supply: int
    decimals: int
    transferable: bool
    mintable: bool
    burnable: bool
    stakeable: bool

    # Metaverse utilities
    metaverse_utilities: List[str] = field(default_factory=list)
    virtual_world_compatibility: List[str] = field(default_factory=list)
    nft_interactions: bool = False

    # Economic properties
    inflation_rate: float = 0.0
    deflationary_mechanism: Optional[str] = None
    vesting_schedule: Optional[Dict[str, Any]] = None

    # Governance properties
    voting_weight: float = 1.0
    proposal_threshold: int = 1000

    # Technical specifications
    quantum_safe: bool = True
    offline_capability: bool = True
    cross_chain_compatible: bool = False


class MetaverseUtilityManager:
    """
    Manages token utilities within metaverse and virtual environments.
    Handles avatar systems, virtual economies, and cross-world interactions.
    """

    def __init__(self, wallet: QuantumWallet, token_rail: TokenRail):
        self.wallet = wallet
        self.token_rail = token_rail
        self.utility_keypair = new_keypair()

        # Initialize token specifications
        self.token_specs = self._initialize_token_specifications()

        # Metaverse state tracking
        self.avatar_registry: Dict[str, Dict] = {}
        self.virtual_worlds: Dict[str, Dict] = {}
        self.asset_marketplace: Dict[str, Dict] = {}
        self.utility_contracts: Dict[str, Dict] = {}

        logger.info("LL TOKEN Metaverse Utility Manager initialized")

    def _initialize_token_specifications(self) -> Dict[TokenType, TokenSpecification]:
        """Initialize all LL TOKEN specifications with metaverse utilities."""

        specs = {
            # Core Utility Tokens
            TokenType.LLT_COMPUTE: TokenSpecification(
                token_type=TokenType.LLT_COMPUTE,
                name="LL TOKEN Compute",
                symbol="LLT-COMPUTE",
                total_supply=100_000_000,
                decimals=6,
                transferable=True,
                mintable=True,
                burnable=True,
                stakeable=True,
                metaverse_utilities=[
                    "AI model training acceleration",
                    "Virtual world physics simulation",
                    "Real-time avatar animation",
                    "Metaverse infrastructure hosting",
                    "Distributed computing marketplace"
                ],
                virtual_world_compatibility=["Unity", "Unreal", "VRChat", "Horizon Worlds", "Decentraland"],
                nft_interactions=True,
                inflation_rate=0.05,  # 5% annual to incentivize compute provision
                voting_weight=1.0
            ),

            TokenType.LLT_DATA: TokenSpecification(
                token_type=TokenType.LLT_DATA,
                name="LL TOKEN Data",
                symbol="LLT-DATA",
                total_supply=250_000_000,
                decimals=6,
                transferable=True,
                mintable=True,
                burnable=False,
                stakeable=False,
                metaverse_utilities=[
                    "Avatar behavior data access",
                    "Virtual world analytics",
                    "User interaction patterns",
                    "Federated learning datasets",
                    "Cross-world identity verification"
                ],
                virtual_world_compatibility=["All major metaverse platforms"],
                nft_interactions=True,
                deflationary_mechanism="Usage-based burning",
                voting_weight=0.5
            ),

            TokenType.LLT_GOVERNANCE: TokenSpecification(
                token_type=TokenType.LLT_GOVERNANCE,
                name="LL TOKEN Governance",
                symbol="LLT-GOV",
                total_supply=10_000_000,
                decimals=6,
                transferable=True,
                mintable=False,
                burnable=False,
                stakeable=True,
                metaverse_utilities=[
                    "Virtual world governance",
                    "Protocol parameter voting",
                    "Metaverse standards development",
                    "Community fund allocation",
                    "Cross-chain bridge governance"
                ],
                virtual_world_compatibility=["Governance applies across all platforms"],
                nft_interactions=False,
                voting_weight=10.0,  # High voting weight
                proposal_threshold=100_000
            ),

            # Metaverse Utility Tokens
            TokenType.LLT_AVATAR: TokenSpecification(
                token_type=TokenType.LLT_AVATAR,
                name="LL TOKEN Avatar",
                symbol="LLT-AVATAR",
                total_supply=1_000_000_000,
                decimals=6,
                transferable=True,
                mintable=True,
                burnable=True,
                stakeable=False,
                metaverse_utilities=[
                    "Avatar customization and upgrades",
                    "Unique avatar abilities and skills",
                    "Cross-world avatar portability",
                    "Avatar-to-avatar interactions",
                    "Virtual identity management",
                    "Avatar marketplace transactions"
                ],
                virtual_world_compatibility=[
                    "VRChat", "Horizon Worlds", "Rec Room", "AltspaceVR",
                    "Mozilla Hubs", "Spatial", "Engage", "Custom Unity/Unreal worlds"
                ],
                nft_interactions=True,
                deflationary_mechanism="Avatar upgrade burning",
                voting_weight=0.1
            ),

            TokenType.LLT_LAND: TokenSpecification(
                token_type=TokenType.LLT_LAND,
                name="LL TOKEN Land",
                symbol="LLT-LAND",
                total_supply=50_000_000,
                decimals=6,
                transferable=True,
                mintable=False,  # Fixed supply for scarcity
                burnable=False,
                stakeable=True,
                metaverse_utilities=[
                    "Virtual land ownership rights",
                    "Land development and construction",
                    "Virtual real estate marketplace",
                    "Land rental and leasing",
                    "Territorial governance rights",
                    "Resource extraction from virtual land"
                ],
                virtual_world_compatibility=[
                    "Decentraland", "The Sandbox", "Somnium Space", "Cryptovoxels",
                    "Earth 2", "SuperWorld", "Custom blockchain worlds"
                ],
                nft_interactions=True,
                voting_weight=5.0,  # Land owners have significant governance weight
                proposal_threshold=10_000
            ),

            TokenType.LLT_ASSET: TokenSpecification(
                token_type=TokenType.LLT_ASSET,
                name="LL TOKEN Asset",
                symbol="LLT-ASSET",
                total_supply=500_000_000,
                decimals=6,
                transferable=True,
                mintable=True,
                burnable=True,
                stakeable=False,
                metaverse_utilities=[
                    "Virtual asset creation and minting",
                    "Asset trading and marketplace fees",
                    "Cross-world asset interoperability",
                    "Digital collectibles and NFTs",
                    "Virtual goods and services",
                    "Asset authentication and verification"
                ],
                virtual_world_compatibility=["Universal - all metaverse platforms"],
                nft_interactions=True,
                inflation_rate=0.03,  # 3% annual to encourage asset creation
                deflationary_mechanism="Trading fee burning",
                voting_weight=0.2
            ),

            TokenType.LLT_EXPERIENCE: TokenSpecification(
                token_type=TokenType.LLT_EXPERIENCE,
                name="LL TOKEN Experience",
                symbol="LLT-EXP",
                total_supply=2_000_000_000,
                decimals=6,
                transferable=False,  # Soul-bound token
                mintable=True,
                burnable=False,
                stakeable=True,
                metaverse_utilities=[
                    "Avatar skill and level progression",
                    "Achievement and milestone rewards",
                    "Experience-gated content access",
                    "Reputation and status systems",
                    "Learning and education rewards",
                    "Cross-world skill recognition"
                ],
                virtual_world_compatibility=["All gaming and educational metaverses"],
                nft_interactions=False,  # Soul-bound
                inflation_rate=0.10,  # 10% annual to reward continuous engagement
                voting_weight=0.0  # No voting rights - utility only
            ),

            # Economic Tokens
            TokenType.LLT_STAKE: TokenSpecification(
                token_type=TokenType.LLT_STAKE,
                name="LL TOKEN Stake",
                symbol="LLT-STAKE",
                total_supply=75_000_000,
                decimals=6,
                transferable=True,
                mintable=True,
                burnable=True,
                stakeable=True,
                metaverse_utilities=[
                    "Validator node operation",
                    "Network security and consensus",
                    "Staking rewards and yield generation",
                    "Slashing protection mechanisms",
                    "Delegate voting rights"
                ],
                virtual_world_compatibility=["Backend infrastructure for all worlds"],
                nft_interactions=False,
                inflation_rate=0.08,  # 8% annual staking rewards
                voting_weight=3.0,
                proposal_threshold=50_000
            ),

            # Social Tokens
            TokenType.LLT_REPUTATION: TokenSpecification(
                token_type=TokenType.LLT_REPUTATION,
                name="LL TOKEN Reputation",
                symbol="LLT-REP",
                total_supply=1_000_000_000,
                decimals=6,
                transferable=False,  # Soul-bound
                mintable=True,
                burnable=True,  # Can be lost through bad behavior
                stakeable=False,
                metaverse_utilities=[
                    "Social status and credibility",
                    "Access to exclusive communities",
                    "Trust scores for transactions",
                    "Moderation and governance privileges",
                    "Reputation-based rewards",
                    "Cross-platform identity verification"
                ],
                virtual_world_compatibility=["All social metaverse platforms"],
                nft_interactions=False,
                deflationary_mechanism="Bad behavior penalties",
                voting_weight=2.0,
                proposal_threshold=500_000
            ),

            TokenType.LLT_COLLABORATION: TokenSpecification(
                token_type=TokenType.LLT_COLLABORATION,
                name="LL TOKEN Collaboration",
                symbol="LLT-COLLAB",
                total_supply=300_000_000,
                decimals=6,
                transferable=True,
                mintable=True,
                burnable=False,
                stakeable=True,
                metaverse_utilities=[
                    "Team formation and project funding",
                    "Collaborative workspace access",
                    "Shared resource management",
                    "Group achievement rewards",
                    "Cross-team collaboration bonuses",
                    "Distributed project governance"
                ],
                virtual_world_compatibility=["Work and collaboration focused metaverses"],
                nft_interactions=True,
                inflation_rate=0.06,  # 6% annual to encourage collaboration
                voting_weight=1.5
            ),

            TokenType.LLT_EDUCATION: TokenSpecification(
                token_type=TokenType.LLT_EDUCATION,
                name="LL TOKEN Education",
                symbol="LLT-EDU",
                total_supply=200_000_000,
                decimals=6,
                transferable=True,
                mintable=True,
                burnable=False,
                stakeable=True,
                metaverse_utilities=[
                    "Educational content access",
                    "Course completion certifications",
                    "Teacher and mentor rewards",
                    "Educational resource creation",
                    "Cross-institutional credit transfer",
                    "Skill verification and badging"
                ],
                virtual_world_compatibility=["Educational metaverses and VR classrooms"],
                nft_interactions=True,  # For certificates and badges
                inflation_rate=0.04,  # 4% annual for educational incentives
                voting_weight=0.3
            ),

            TokenType.LLT_CREATOR: TokenSpecification(
                token_type=TokenType.LLT_CREATOR,
                name="LL TOKEN Creator",
                symbol="LLT-CREATE",
                total_supply=400_000_000,
                decimals=6,
                transferable=True,
                mintable=True,
                burnable=True,
                stakeable=True,
                metaverse_utilities=[
                    "Content creation tools access",
                    "Creator monetization and royalties",
                    "Collaborative creation projects",
                    "Content distribution rights",
                    "Creator support and funding",
                    "Intellectual property protection"
                ],
                virtual_world_compatibility=["All content creation platforms"],
                nft_interactions=True,
                inflation_rate=0.07,  # 7% annual to support creators
                deflationary_mechanism="Content quality-based burning",
                voting_weight=1.2
            )
        }

        return specs

    def get_token_specifications(self) -> Dict[str, Dict]:
        """Get complete token specifications for all LL TOKEN types."""
        return {
            token_type.value: asdict(spec)
            for token_type, spec in self.token_specs.items()
        }

    def create_avatar_utility_contract(
        self,
        avatar_id: str,
        owner_wallet: str,
        initial_attributes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create avatar utility contract for metaverse interactions."""

        contract_id = f"AVATAR-{uuid.uuid4().hex[:12].upper()}"

        avatar_contract = {
            "contract_id": contract_id,
            "avatar_id": avatar_id,
            "owner_wallet": owner_wallet,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "contract_type": "AVATAR_UTILITY",

            # Avatar specifications
            "attributes": {
                "base_stats": initial_attributes.get("stats", {}),
                "appearance": initial_attributes.get("appearance", {}),
                "abilities": initial_attributes.get("abilities", []),
                "inventory": initial_attributes.get("inventory", []),
                "achievements": initial_attributes.get("achievements", [])
            },

            # Token requirements and utilities
            "token_requirements": {
                TokenType.LLT_AVATAR.value: 1000,  # Base avatar creation cost
                TokenType.LLT_EXPERIENCE.value: 0   # Starting experience
            },

            # Metaverse compatibility
            "supported_worlds": self.token_specs[TokenType.LLT_AVATAR].virtual_world_compatibility,
            "cross_world_enabled": True,

            # Upgrade mechanics
            "upgrade_costs": {
                "appearance_slot": {TokenType.LLT_AVATAR.value: 500},
                "ability_unlock": {TokenType.LLT_AVATAR.value: 1000, TokenType.LLT_EXPERIENCE.value: 5000},
                "stat_boost": {TokenType.LLT_AVATAR.value: 200}
            },

            # Revenue sharing
            "creator_royalty": 0.05,  # 5% royalty to original creator
            "platform_fee": 0.02,     # 2% platform fee

            # Status
            "active": True,
            "quantum_signed": True
        }

        # Create cryptographic proof
        contract_bytes = json.dumps(avatar_contract, sort_keys=True).encode()
        signature = sign_blob(self.utility_keypair[0], contract_bytes)
        avatar_contract["signature"] = signature.hex()

        # Register avatar
        self.avatar_registry[avatar_id] = avatar_contract

        logger.info(f"Created avatar utility contract: {contract_id} for avatar {avatar_id}")
        return avatar_contract

    def create_virtual_world_economy(
        self,
        world_id: str,
        world_name: str,
        supported_tokens: List[TokenType],
        economic_model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create economic framework for a virtual world."""

        economy_id = f"ECONOMY-{uuid.uuid4().hex[:12].upper()}"

        world_economy = {
            "economy_id": economy_id,
            "world_id": world_id,
            "world_name": world_name,
            "created_at": datetime.now(timezone.utc).isoformat(),

            # Supported token types and their utilities
            "supported_tokens": {
                token_type.value: {
                    "utility_functions": self.token_specs[token_type].metaverse_utilities,
                    "exchange_rate": economic_model.get("exchange_rates", {}).get(token_type.value, 1.0),
                    "daily_limits": economic_model.get("daily_limits", {}).get(token_type.value, 10000)
                }
                for token_type in supported_tokens
            },

            # Economic parameters
            "economic_model": {
                "base_currency": economic_model.get("base_currency", TokenType.LLT_AVATAR.value),
                "inflation_rate": economic_model.get("inflation_rate", 0.03),
                "transaction_fees": economic_model.get("transaction_fees", 0.001),
                "validator_rewards": economic_model.get("validator_rewards", 0.05),
                "creator_incentives": economic_model.get("creator_incentives", 0.10)
            },

            # Marketplace mechanics
            "marketplace": {
                "asset_trading_enabled": True,
                "nft_support": True,
                "cross_world_trading": economic_model.get("cross_world_trading", True),
                "automated_market_maker": economic_model.get("amm_enabled", True)
            },

            # Governance structure
            "governance": {
                "voting_tokens": [TokenType.LLT_GOVERNANCE.value, TokenType.LLT_LAND.value],
                "proposal_threshold": 10000,
                "voting_period_days": 7,
                "execution_delay_days": 2
            },

            # Integration APIs
            "api_endpoints": {
                "token_balance": f"/api/v1/worlds/{world_id}/balance",
                "transfer": f"/api/v1/worlds/{world_id}/transfer",
                "marketplace": f"/api/v1/worlds/{world_id}/marketplace",
                "governance": f"/api/v1/worlds/{world_id}/governance"
            },

            # Status
            "active": True,
            "quantum_secured": True
        }

        # Register world economy
        self.virtual_worlds[world_id] = world_economy

        logger.info(f"Created virtual world economy: {economy_id} for world {world_name}")
        return world_economy

    def create_cross_world_bridge(
        self,
        source_world: str,
        target_world: str,
        bridgeable_tokens: List[TokenType],
        bridge_parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create cross-world token bridge for metaverse interoperability."""

        bridge_id = f"BRIDGE-{uuid.uuid4().hex[:12].upper()}"
        bridge_params = bridge_parameters or {}

        bridge_contract = {
            "bridge_id": bridge_id,
            "source_world": source_world,
            "target_world": target_world,
            "created_at": datetime.now(timezone.utc).isoformat(),

            # Bridgeable assets
            "bridgeable_tokens": {
                token_type.value: {
                    "enabled": True,
                    "transfer_fee": bridge_params.get("transfer_fees", {}).get(token_type.value, 0.001),
                    "daily_limit": bridge_params.get("daily_limits", {}).get(token_type.value, 100000),
                    "confirmation_blocks": bridge_params.get("confirmations", 6)
                }
                for token_type in bridgeable_tokens
            },

            # Bridge mechanics
            "bridge_mechanics": {
                "lock_and_mint": bridge_params.get("lock_and_mint", True),
                "burn_and_mint": bridge_params.get("burn_and_mint", False),
                "native_wrapping": bridge_params.get("native_wrapping", True),
                "atomic_swaps": bridge_params.get("atomic_swaps", True)
            },

            # Security parameters
            "security": {
                "multi_sig_required": True,
                "validator_threshold": bridge_params.get("validator_threshold", 3),
                "challenge_period_hours": bridge_params.get("challenge_period", 24),
                "quantum_safe_signatures": True
            },

            # Fee distribution
            "fee_distribution": {
                "validators": 0.6,
                "bridge_maintenance": 0.2,
                "insurance_fund": 0.1,
                "protocol_treasury": 0.1
            },

            # Monitoring and analytics
            "metrics": {
                "total_volume": 0,
                "total_transactions": 0,
                "average_transfer_time": "10 minutes",
                "success_rate": 0.999
            },

            # Status
            "active": True,
            "quantum_secured": True
        }

        logger.info(f"Created cross-world bridge: {bridge_id} ({source_world} <-> {target_world})")
        return bridge_contract

    def calculate_token_utility_value(
        self,
        token_type: TokenType,
        amount: int,
        utility_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate the utility value of tokens in specific metaverse contexts."""

        spec = self.token_specs[token_type]
        context_world = utility_context.get("world_id", "generic")

        utility_calculation = {
            "token_type": token_type.value,
            "amount": amount,
            "context": utility_context,
            "calculated_at": datetime.now(timezone.utc).isoformat(),

            # Base utility value
            "base_value": amount * utility_context.get("base_rate", 1.0),

            # Context multipliers
            "multipliers": {
                "world_compatibility": 1.0,
                "utility_demand": 1.0,
                "staking_bonus": 1.0,
                "reputation_bonus": 1.0
            },

            # Specific utilities enabled
            "enabled_utilities": [],

            # Total calculated value
            "total_utility_value": 0
        }

        # Check world compatibility
        if context_world in spec.virtual_world_compatibility or "All" in spec.virtual_world_compatibility[0]:
            utility_calculation["multipliers"]["world_compatibility"] = 1.2

        # Calculate utility-specific values
        for utility in spec.metaverse_utilities:
            utility_value = self._calculate_individual_utility_value(
                utility, amount, utility_context
            )
            if utility_value > 0:
                utility_calculation["enabled_utilities"].append({
                    "utility": utility,
                    "value": utility_value
                })

        # Apply multipliers
        base_value = utility_calculation["base_value"]
        for multiplier in utility_calculation["multipliers"].values():
            base_value *= multiplier

        utility_calculation["total_utility_value"] = base_value

        return utility_calculation

    def _calculate_individual_utility_value(
        self,
        utility: str,
        amount: int,
        context: Dict[str, Any]
    ) -> float:
        """Calculate value for individual utility function."""

        # Utility-specific calculations
        utility_rates = {
            "Avatar customization and upgrades": amount * 0.1,
            "Virtual land ownership rights": amount * 0.5,
            "AI model training acceleration": amount * 0.2,
            "Educational content access": amount * 0.05,
            "Creator monetization and royalties": amount * 0.15,
            "Virtual world governance": amount * 1.0,
            "Asset trading and marketplace fees": amount * 0.02
        }

        return utility_rates.get(utility, amount * 0.01)

    def generate_metaverse_compatibility_report(self) -> Dict[str, Any]:
        """Generate comprehensive metaverse compatibility report."""

        report_id = f"METAVERSE-COMPAT-{uuid.uuid4().hex[:12].upper()}"

        compatibility_report = {
            "report_id": report_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ll_token_version": "1.0.0",
            "quantum_safe": True,

            # Token compatibility matrix
            "token_compatibility": {},

            # Supported metaverse platforms
            "supported_platforms": set(),

            # Cross-chain capabilities
            "cross_chain_support": {
                "ethereum": False,
                "polygon": False,
                "binance_smart_chain": False,
                "solana": False,
                "quantum_native": True
            },

            # Feature matrix
            "feature_matrix": {
                "nft_interactions": sum(1 for spec in self.token_specs.values() if spec.nft_interactions),
                "stakeable_tokens": sum(1 for spec in self.token_specs.values() if spec.stakeable),
                "soul_bound_tokens": sum(1 for spec in self.token_specs.values() if not spec.transferable),
                "governance_tokens": sum(1 for spec in self.token_specs.values() if spec.voting_weight > 1.0),
                "utility_tokens": len(self.token_specs)
            },

            # Economic totals
            "economic_overview": {
                "total_token_types": len(self.token_specs),
                "total_supply_all_tokens": sum(spec.total_supply for spec in self.token_specs.values()),
                "average_inflation_rate": sum(spec.inflation_rate for spec in self.token_specs.values()) / len(self.token_specs),
                "deflationary_tokens": sum(1 for spec in self.token_specs.values() if spec.deflationary_mechanism)
            }
        }

        # Build compatibility matrix
        for token_type, spec in self.token_specs.items():
            compatibility_report["token_compatibility"][token_type.value] = {
                "name": spec.name,
                "utilities": len(spec.metaverse_utilities),
                "supported_worlds": len(spec.virtual_world_compatibility),
                "nft_compatible": spec.nft_interactions,
                "cross_world_portable": True
            }

            # Collect all supported platforms
            compatibility_report["supported_platforms"].update(spec.virtual_world_compatibility)

        # Convert set to list for JSON serialization
        compatibility_report["supported_platforms"] = list(compatibility_report["supported_platforms"])

        logger.info(f"Generated metaverse compatibility report: {report_id}")
        return compatibility_report


def create_comprehensive_token_economy(
    wallet: QuantumWallet,
    token_rail: TokenRail
) -> Tuple[MetaverseUtilityManager, Dict[str, Any]]:
    """
    Create complete LL TOKEN metaverse economy with all token specifications.

    Returns:
        - MetaverseUtilityManager instance
        - Complete token specifications dictionary
    """

    # Create utility manager
    utility_manager = MetaverseUtilityManager(wallet, token_rail)

    # Get complete token specifications
    token_specifications = utility_manager.get_token_specifications()

    # Generate compatibility report
    compatibility_report = utility_manager.generate_metaverse_compatibility_report()

    logger.info("✅ Comprehensive LL TOKEN metaverse economy created")
    logger.info(f"Token types: {len(token_specifications)}")
    logger.info(f"Supported platforms: {len(compatibility_report['supported_platforms'])}")
    logger.info(f"Total token supply: {compatibility_report['economic_overview']['total_supply_all_tokens']:,}")

    return utility_manager, {
        "token_specifications": token_specifications,
        "compatibility_report": compatibility_report,
        "economic_overview": compatibility_report["economic_overview"]
    }