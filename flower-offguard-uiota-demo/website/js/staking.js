// LL TOKEN OFFLINE - Staking Module
// Quantum-safe staking and yield farming for LL TOKEN ecosystem

window.LLTokenStaking = {
    // Staking pools configuration
    stakingPools: {
        'LLT-COMPUTE': {
            name: 'Compute Staking Pool',
            apy: 12.5,
            lockPeriod: 30, // days
            minStake: 100,
            maxStake: 10000,
            totalStaked: 2500000,
            poolCapacity: 5000000,
            rewardToken: 'LLT-COMPUTE',
            description: 'Stake compute tokens to earn rewards from federated learning tasks'
        },
        'LLT-DATA': {
            name: 'Data Quality Pool',
            apy: 15.0,
            lockPeriod: 60,
            minStake: 50,
            maxStake: 5000,
            totalStaked: 1800000,
            poolCapacity: 3000000,
            rewardToken: 'LLT-DATA',
            description: 'Stake data tokens and earn from data quality validation'
        },
        'LLT-GOV': {
            name: 'Governance Staking',
            apy: 8.0,
            lockPeriod: 90,
            minStake: 1000,
            maxStake: 50000,
            totalStaked: 5200000,
            poolCapacity: 8000000,
            rewardToken: 'LLT-GOV',
            description: 'Stake governance tokens and participate in protocol decisions'
        },
        'LLT-AVATAR': {
            name: 'Metaverse Avatar Pool',
            apy: 18.0,
            lockPeriod: 14,
            minStake: 25,
            maxStake: 2000,
            totalStaked: 850000,
            poolCapacity: 1500000,
            rewardToken: 'LLT-AVATAR',
            description: 'Stake avatar tokens and earn from metaverse interactions'
        },
        'LLT-LAND': {
            name: 'Virtual Land Staking',
            apy: 22.0,
            lockPeriod: 180,
            minStake: 500,
            maxStake: 20000,
            totalStaked: 3400000,
            poolCapacity: 6000000,
            rewardToken: 'LLT-LAND',
            description: 'Stake virtual land tokens for premium location rewards'
        }
    },

    // User staking positions
    userPositions: {},

    // Initialize staking system
    init: function() {
        this.loadUserPositions();
        this.startRewardCalculation();
        this.bindEvents();
        console.log('🏦 Staking system initialized');
    },

    // Load user staking positions from storage
    loadUserPositions: function() {
        const stored = localStorage.getItem('lltoken_staking_positions');
        if (stored) {
            this.userPositions = JSON.parse(stored);
        }
    },

    // Save user positions to storage
    saveUserPositions: function() {
        localStorage.setItem('lltoken_staking_positions', JSON.stringify(this.userPositions));
    },

    // Stake tokens in a pool
    async stakeTokens(poolId, amount, lockPeriod = null) {
        const pool = this.stakingPools[poolId];
        if (!pool) {
            throw new Error('Invalid staking pool');
        }

        // Validate staking amount
        if (amount < pool.minStake || amount > pool.maxStake) {
            throw new Error(`Stake amount must be between ${pool.minStake} and ${pool.maxStake} ${poolId}`);
        }

        // Check pool capacity
        if (pool.totalStaked + amount > pool.poolCapacity) {
            throw new Error('Pool capacity exceeded');
        }

        // Check user balance
        const userBalance = window.LLTokenWallet.getTokenBalance(poolId);
        if (userBalance < amount) {
            throw new Error('Insufficient token balance');
        }

        // Generate staking transaction
        const stakingId = this.generateStakingId();
        const timestamp = Date.now();
        const unlockTime = timestamp + (lockPeriod || pool.lockPeriod) * 24 * 60 * 60 * 1000;

        const stakingPosition = {
            id: stakingId,
            poolId: poolId,
            amount: amount,
            apy: pool.apy,
            stakedAt: timestamp,
            unlockTime: unlockTime,
            lockPeriod: lockPeriod || pool.lockPeriod,
            rewardsEarned: 0,
            lastRewardCalculation: timestamp,
            status: 'active',
            autoReinvest: false
        };

        // Record staking transaction
        await window.LLTokenWallet.transferTokens(
            window.LLTokenWallet.getWalletAddress(),
            `staking_pool_${poolId}`,
            amount,
            poolId,
            {
                type: 'staking_deposit',
                poolId: poolId,
                stakingId: stakingId,
                lockPeriod: stakingPosition.lockPeriod,
                expectedApy: pool.apy
            }
        );

        // Update user positions
        if (!this.userPositions[poolId]) {
            this.userPositions[poolId] = [];
        }
        this.userPositions[poolId].push(stakingPosition);

        // Update pool stats
        pool.totalStaked += amount;

        this.saveUserPositions();
        this.updateStakingUI();

        console.log(`💰 Staked ${amount} ${poolId} in ${pool.name}`);
        this.showToast(`Successfully staked ${amount} ${poolId}`, 'success');

        return stakingPosition;
    },

    // Unstake tokens from a pool
    async unstakeTokens(stakingId, poolId) {
        const positions = this.userPositions[poolId];
        if (!positions) {
            throw new Error('No staking positions found');
        }

        const positionIndex = positions.findIndex(pos => pos.id === stakingId);
        if (positionIndex === -1) {
            throw new Error('Staking position not found');
        }

        const position = positions[positionIndex];
        const currentTime = Date.now();

        // Check if lock period has expired
        if (currentTime < position.unlockTime && position.status === 'active') {
            const remainingDays = Math.ceil((position.unlockTime - currentTime) / (24 * 60 * 60 * 1000));
            throw new Error(`Position is still locked for ${remainingDays} days`);
        }

        // Calculate final rewards
        this.calculateRewards(position);

        // Record unstaking transaction
        const totalReturn = position.amount + position.rewardsEarned;
        await window.LLTokenWallet.transferTokens(
            `staking_pool_${poolId}`,
            window.LLTokenWallet.getWalletAddress(),
            totalReturn,
            poolId,
            {
                type: 'staking_withdrawal',
                stakingId: stakingId,
                principalAmount: position.amount,
                rewardAmount: position.rewardsEarned,
                stakingDuration: Math.floor((currentTime - position.stakedAt) / (24 * 60 * 60 * 1000))
            }
        );

        // Update pool stats
        const pool = this.stakingPools[poolId];
        pool.totalStaked -= position.amount;

        // Remove position
        positions.splice(positionIndex, 1);
        if (positions.length === 0) {
            delete this.userPositions[poolId];
        }

        this.saveUserPositions();
        this.updateStakingUI();

        console.log(`💸 Unstaked ${totalReturn} ${poolId} (${position.amount} principal + ${position.rewardsEarned} rewards)`);
        this.showToast(`Unstaked ${totalReturn} ${poolId} successfully`, 'success');

        return {
            principal: position.amount,
            rewards: position.rewardsEarned,
            total: totalReturn
        };
    },

    // Calculate rewards for a staking position
    calculateRewards: function(position) {
        const currentTime = Date.now();
        const timeElapsed = currentTime - position.lastRewardCalculation;
        const hoursElapsed = timeElapsed / (1000 * 60 * 60);

        // Calculate rewards based on APY
        const hourlyReward = (position.amount * (position.apy / 100)) / (365 * 24);
        const newRewards = hourlyReward * hoursElapsed;

        position.rewardsEarned += newRewards;
        position.lastRewardCalculation = currentTime;

        // Auto-compound if enabled
        if (position.autoReinvest && newRewards > 0) {
            position.amount += newRewards;
            position.rewardsEarned = 0;
            console.log(`🔄 Auto-compounded ${newRewards.toFixed(6)} ${position.poolId}`);
        }

        return newRewards;
    },

    // Start reward calculation timer
    startRewardCalculation: function() {
        setInterval(() => {
            let totalNewRewards = 0;

            Object.keys(this.userPositions).forEach(poolId => {
                this.userPositions[poolId].forEach(position => {
                    if (position.status === 'active') {
                        const newRewards = this.calculateRewards(position);
                        totalNewRewards += newRewards;
                    }
                });
            });

            if (totalNewRewards > 0) {
                this.saveUserPositions();
                this.updateStakingUI();
            }
        }, 60000); // Update every minute
    },

    // Toggle auto-reinvestment for a position
    toggleAutoReinvest: function(stakingId, poolId) {
        const positions = this.userPositions[poolId];
        if (positions) {
            const position = positions.find(pos => pos.id === stakingId);
            if (position) {
                position.autoReinvest = !position.autoReinvest;
                this.saveUserPositions();
                this.updateStakingUI();

                const status = position.autoReinvest ? 'enabled' : 'disabled';
                this.showToast(`Auto-reinvest ${status} for ${poolId} position`, 'success');
            }
        }
    },

    // Get user's total staked value across all pools
    getTotalStakedValue: function() {
        let totalValue = 0;

        Object.keys(this.userPositions).forEach(poolId => {
            this.userPositions[poolId].forEach(position => {
                totalValue += position.amount + position.rewardsEarned;
            });
        });

        return totalValue;
    },

    // Get user's total pending rewards
    getTotalPendingRewards: function() {
        let totalRewards = 0;

        Object.keys(this.userPositions).forEach(poolId => {
            this.userPositions[poolId].forEach(position => {
                // Calculate current rewards
                this.calculateRewards(position);
                totalRewards += position.rewardsEarned;
            });
        });

        return totalRewards;
    },

    // Emergency unstake (with penalty)
    async emergencyUnstake(stakingId, poolId) {
        const positions = this.userPositions[poolId];
        if (!positions) {
            throw new Error('No staking positions found');
        }

        const positionIndex = positions.findIndex(pos => pos.id === stakingId);
        if (positionIndex === -1) {
            throw new Error('Staking position not found');
        }

        const position = positions[positionIndex];
        const currentTime = Date.now();

        // Calculate penalty (10% of principal if unstaking early)
        const penalty = position.unlockTime > currentTime ? position.amount * 0.1 : 0;
        const returnAmount = position.amount - penalty + position.rewardsEarned;

        // Record emergency unstaking transaction
        await window.LLTokenWallet.transferTokens(
            `staking_pool_${poolId}`,
            window.LLTokenWallet.getWalletAddress(),
            returnAmount,
            poolId,
            {
                type: 'emergency_unstaking',
                stakingId: stakingId,
                principalAmount: position.amount,
                rewardAmount: position.rewardsEarned,
                penaltyAmount: penalty,
                earlyWithdrawal: position.unlockTime > currentTime
            }
        );

        // Update pool stats
        const pool = this.stakingPools[poolId];
        pool.totalStaked -= position.amount;

        // Remove position
        positions.splice(positionIndex, 1);
        if (positions.length === 0) {
            delete this.userPositions[poolId];
        }

        this.saveUserPositions();
        this.updateStakingUI();

        console.log(`⚠️ Emergency unstaked ${returnAmount} ${poolId} (penalty: ${penalty})`);
        this.showToast(`Emergency unstake completed. Penalty: ${penalty.toFixed(2)} ${poolId}`, 'warning');

        return {
            principal: position.amount,
            rewards: position.rewardsEarned,
            penalty: penalty,
            total: returnAmount
        };
    },

    // Generate unique staking ID
    generateStakingId: function() {
        return 'stake_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    },

    // Update staking UI elements
    updateStakingUI: function() {
        // Update staking overview
        this.updateStakingOverview();

        // Update active positions
        this.updateActivePositions();

        // Update staking pools display
        this.updateStakingPools();
    },

    // Update staking overview statistics
    updateStakingOverview: function() {
        const totalStaked = this.getTotalStakedValue();
        const totalRewards = this.getTotalPendingRewards();
        const activePositions = Object.values(this.userPositions).flat().length;

        // Update DOM elements if they exist
        const totalStakedEl = document.getElementById('total-staked-value');
        const totalRewardsEl = document.getElementById('total-pending-rewards');
        const activePositionsEl = document.getElementById('active-positions-count');

        if (totalStakedEl) totalStakedEl.textContent = totalStaked.toFixed(2);
        if (totalRewardsEl) totalRewardsEl.textContent = totalRewards.toFixed(6);
        if (activePositionsEl) activePositionsEl.textContent = activePositions;
    },

    // Update active positions display
    updateActivePositions: function() {
        const positionsContainer = document.getElementById('active-staking-positions');
        if (!positionsContainer) return;

        let positionsHTML = '';

        Object.keys(this.userPositions).forEach(poolId => {
            const pool = this.stakingPools[poolId];
            this.userPositions[poolId].forEach(position => {
                const currentTime = Date.now();
                const isLocked = currentTime < position.unlockTime;
                const remainingDays = isLocked ?
                    Math.ceil((position.unlockTime - currentTime) / (24 * 60 * 60 * 1000)) : 0;

                positionsHTML += `
                    <div class="bg-white/5 backdrop-blur-sm border border-purple-500/20 rounded-xl p-4 mb-4">
                        <div class="flex justify-between items-start mb-3">
                            <div>
                                <h4 class="text-lg font-semibold text-white">${pool.name}</h4>
                                <p class="text-sm text-gray-400">${position.amount} ${poolId}</p>
                            </div>
                            <div class="text-right">
                                <div class="text-sm text-green-400">${position.apy}% APY</div>
                                <div class="text-xs text-gray-400">${isLocked ? `${remainingDays} days locked` : 'Unlocked'}</div>
                            </div>
                        </div>

                        <div class="grid grid-cols-2 gap-4 mb-3">
                            <div>
                                <div class="text-xs text-gray-400">Rewards Earned</div>
                                <div class="text-sm font-medium text-green-300">${position.rewardsEarned.toFixed(6)}</div>
                            </div>
                            <div>
                                <div class="text-xs text-gray-400">Total Value</div>
                                <div class="text-sm font-medium text-white">${(position.amount + position.rewardsEarned).toFixed(2)}</div>
                            </div>
                        </div>

                        <div class="flex space-x-2">
                            <button onclick="LLTokenStaking.toggleAutoReinvest('${position.id}', '${poolId}')"
                                    class="px-3 py-1 text-xs rounded-lg border border-purple-500/30 transition-all
                                           ${position.autoReinvest ? 'bg-purple-500/20 text-purple-300' : 'bg-transparent text-gray-400 hover:bg-purple-500/10'}">
                                Auto-Reinvest ${position.autoReinvest ? 'ON' : 'OFF'}
                            </button>

                            ${!isLocked ? `
                                <button onclick="LLTokenStaking.unstakeTokens('${position.id}', '${poolId}')"
                                        class="px-3 py-1 text-xs bg-green-500/20 text-green-400 rounded-lg border border-green-500/30 hover:bg-green-500/30 transition-all">
                                    Unstake
                                </button>
                            ` : `
                                <button onclick="LLTokenStaking.emergencyUnstake('${position.id}', '${poolId}')"
                                        class="px-3 py-1 text-xs bg-red-500/20 text-red-400 rounded-lg border border-red-500/30 hover:bg-red-500/30 transition-all">
                                    Emergency Unstake
                                </button>
                            `}
                        </div>
                    </div>
                `;
            });
        });

        positionsContainer.innerHTML = positionsHTML || '<div class="text-center text-gray-400 py-8">No active staking positions</div>';
    },

    // Update staking pools display
    updateStakingPools: function() {
        const poolsContainer = document.getElementById('staking-pools-list');
        if (!poolsContainer) return;

        let poolsHTML = '';

        Object.keys(this.stakingPools).forEach(poolId => {
            const pool = this.stakingPools[poolId];
            const utilization = (pool.totalStaked / pool.poolCapacity * 100).toFixed(1);

            poolsHTML += `
                <div class="bg-white/5 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6 hover:bg-white/8 transition-all">
                    <div class="flex justify-between items-start mb-4">
                        <div>
                            <h3 class="text-xl font-semibold text-white mb-1">${pool.name}</h3>
                            <p class="text-sm text-gray-400">${pool.description}</p>
                        </div>
                        <div class="text-right">
                            <div class="text-2xl font-bold text-green-400">${pool.apy}%</div>
                            <div class="text-xs text-gray-400">APY</div>
                        </div>
                    </div>

                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                        <div>
                            <div class="text-xs text-gray-400">Min Stake</div>
                            <div class="text-sm font-medium text-white">${pool.minStake} ${poolId}</div>
                        </div>
                        <div>
                            <div class="text-xs text-gray-400">Lock Period</div>
                            <div class="text-sm font-medium text-white">${pool.lockPeriod} days</div>
                        </div>
                        <div>
                            <div class="text-xs text-gray-400">Total Staked</div>
                            <div class="text-sm font-medium text-white">${pool.totalStaked.toLocaleString()}</div>
                        </div>
                        <div>
                            <div class="text-xs text-gray-400">Utilization</div>
                            <div class="text-sm font-medium text-white">${utilization}%</div>
                        </div>
                    </div>

                    <div class="progress-bar mb-4">
                        <div class="progress-fill from-purple-500 to-pink-500" style="width: ${utilization}%"></div>
                    </div>

                    <button onclick="showStakeModal('${poolId}')"
                            class="w-full px-4 py-2 bg-purple-500/20 text-purple-400 rounded-lg border border-purple-500/30 hover:bg-purple-500/30 transition-all">
                        Stake ${poolId}
                    </button>
                </div>
            `;
        });

        poolsContainer.innerHTML = poolsHTML;
    },

    // Bind UI events
    bindEvents: function() {
        // Stake form submission
        const stakeForm = document.getElementById('stake-form');
        if (stakeForm) {
            stakeForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData(stakeForm);
                const poolId = formData.get('poolId');
                const amount = parseFloat(formData.get('amount'));
                const lockPeriod = parseInt(formData.get('lockPeriod'));

                try {
                    await this.stakeTokens(poolId, amount, lockPeriod);
                    stakeForm.reset();
                } catch (error) {
                    console.error('Staking error:', error);
                    this.showToast(error.message, 'error');
                }
            });
        }
    },

    // Show toast notification
    showToast: function(message, type = 'info') {
        if (window.LLToken && window.LLToken.showToast) {
            window.LLToken.showToast(message, type);
        } else {
            console.log(`🔔 ${type.toUpperCase()}: ${message}`);
        }
    }
};

// Global function for stake modal
function showStakeModal(poolId) {
    const pool = window.LLTokenStaking.stakingPools[poolId];
    if (!pool) return;

    const modalHTML = `
        <div id="stake-modal" class="fixed inset-0 bg-black/50 backdrop-blur-lg z-50 flex items-center justify-center">
            <div class="bg-black/80 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6 max-w-md w-full mx-4">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-xl font-semibold text-white">Stake ${poolId}</h3>
                    <button onclick="closeStakeModal()" class="text-gray-400 hover:text-white">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                        </svg>
                    </button>
                </div>

                <form id="stake-modal-form">
                    <input type="hidden" name="poolId" value="${poolId}">

                    <div class="mb-4">
                        <label class="block text-sm text-gray-400 mb-2">Amount to Stake</label>
                        <input type="number" name="amount" min="${pool.minStake}" max="${pool.maxStake}"
                               placeholder="${pool.minStake}" required
                               class="w-full px-3 py-2 bg-white/10 border border-purple-500/30 rounded-lg text-white">
                        <div class="text-xs text-gray-400 mt-1">Min: ${pool.minStake} | Max: ${pool.maxStake} ${poolId}</div>
                    </div>

                    <div class="mb-4">
                        <label class="block text-sm text-gray-400 mb-2">Lock Period (days)</label>
                        <select name="lockPeriod" class="w-full px-3 py-2 bg-white/10 border border-purple-500/30 rounded-lg text-white">
                            <option value="${pool.lockPeriod}">${pool.lockPeriod} days (${pool.apy}% APY)</option>
                            <option value="${pool.lockPeriod * 2}">${pool.lockPeriod * 2} days (${(pool.apy * 1.2).toFixed(1)}% APY)</option>
                            <option value="${pool.lockPeriod * 4}">${pool.lockPeriod * 4} days (${(pool.apy * 1.5).toFixed(1)}% APY)</option>
                        </select>
                    </div>

                    <div class="bg-purple-500/10 rounded-lg p-3 mb-4">
                        <div class="text-sm text-purple-300 mb-2">Pool Information:</div>
                        <div class="text-xs text-gray-400 space-y-1">
                            <div>${pool.description}</div>
                            <div>Pool Utilization: ${(pool.totalStaked / pool.poolCapacity * 100).toFixed(1)}%</div>
                        </div>
                    </div>

                    <div class="flex space-x-3">
                        <button type="button" onclick="closeStakeModal()"
                                class="flex-1 px-4 py-2 bg-gray-500/20 text-gray-400 rounded-lg hover:bg-gray-500/30 transition-all">
                            Cancel
                        </button>
                        <button type="submit"
                                class="flex-1 px-4 py-2 bg-purple-500/20 text-purple-400 rounded-lg border border-purple-500/30 hover:bg-purple-500/30 transition-all">
                            Stake Tokens
                        </button>
                    </div>
                </form>
            </div>
        </div>
    `;

    document.body.insertAdjacentHTML('beforeend', modalHTML);

    // Handle form submission
    document.getElementById('stake-modal-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(e.target);
        const poolId = formData.get('poolId');
        const amount = parseFloat(formData.get('amount'));
        const lockPeriod = parseInt(formData.get('lockPeriod'));

        try {
            await window.LLTokenStaking.stakeTokens(poolId, amount, lockPeriod);
            closeStakeModal();
        } catch (error) {
            console.error('Staking error:', error);
            window.LLTokenStaking.showToast(error.message, 'error');
        }
    });
}

// Global function to close stake modal
function closeStakeModal() {
    const modal = document.getElementById('stake-modal');
    if (modal) {
        modal.remove();
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (window.LLToken && window.LLToken.isOfflineMode) {
        window.LLTokenStaking.init();
    }
});

console.log('🏦 LL TOKEN Staking Module loaded successfully');