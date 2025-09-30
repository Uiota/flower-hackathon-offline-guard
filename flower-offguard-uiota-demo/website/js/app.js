/**
 * LL TOKEN OFFLINE - Main Application JavaScript
 * Quantum-safe wallet and marketplace functionality
 */

// Global Application State
window.LLToken = {
    // Application state
    currentTab: 'wallet',
    isOfflineMode: true,
    walletConnected: false,

    // User data
    walletId: null,
    totalBalance: 0,
    tokenHoldings: {},
    transactions: [],

    // Configuration
    config: {
        apiEndpoint: null, // Offline mode - no API calls
        quantumSafe: true,
        offlineOnly: true
    },

    // Token specifications from the LL TOKEN system
    tokenSpecs: {
        'LLT-COMPUTE': {
            name: 'LL TOKEN Compute',
            symbol: 'LLT-COMPUTE',
            decimals: 6,
            color: '#8b5cf6',
            utilities: ['AI model training', 'Physics simulation', 'Avatar animation']
        },
        'LLT-DATA': {
            name: 'LL TOKEN Data',
            symbol: 'LLT-DATA',
            decimals: 6,
            color: '#06b6d4',
            utilities: ['Avatar behavior data', 'Virtual world analytics', 'FL datasets']
        },
        'LLT-GOV': {
            name: 'LL TOKEN Governance',
            symbol: 'LLT-GOV',
            decimals: 6,
            color: '#f59e0b',
            utilities: ['Virtual world governance', 'Protocol voting', 'Standards development']
        },
        'LLT-AVATAR': {
            name: 'LL TOKEN Avatar',
            symbol: 'LLT-AVATAR',
            decimals: 6,
            color: '#ec4899',
            utilities: ['Avatar customization', 'Abilities & skills', 'Cross-world portability']
        },
        'LLT-LAND': {
            name: 'LL TOKEN Land',
            symbol: 'LLT-LAND',
            decimals: 6,
            color: '#84cc16',
            utilities: ['Virtual land ownership', 'Development rights', 'Real estate marketplace']
        },
        'LLT-ASSET': {
            name: 'LL TOKEN Asset',
            symbol: 'LLT-ASSET',
            decimals: 6,
            color: '#f97316',
            utilities: ['Asset creation & minting', 'Trading marketplace', 'Cross-world interoperability']
        },
        'LLT-EXP': {
            name: 'LL TOKEN Experience',
            symbol: 'LLT-EXP',
            decimals: 6,
            color: '#10b981',
            utilities: ['Skill progression', 'Achievement rewards', 'Reputation systems']
        },
        'LLT-STAKE': {
            name: 'LL TOKEN Stake',
            symbol: 'LLT-STAKE',
            decimals: 6,
            color: '#6366f1',
            utilities: ['Network validation', 'Staking rewards', 'Consensus participation']
        },
        'LLT-REP': {
            name: 'LL TOKEN Reputation',
            symbol: 'LLT-REP',
            decimals: 6,
            color: '#8b5cf6',
            utilities: ['Social status', 'Trust scores', 'Exclusive access']
        },
        'LLT-CREATE': {
            name: 'LL TOKEN Creator',
            symbol: 'LLT-CREATE',
            decimals: 6,
            color: '#ef4444',
            utilities: ['Content creation', 'Creator monetization', 'IP protection']
        }
    }
};

// Initialize Application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🪙 LL TOKEN OFFLINE - Initializing...');

    // Initialize wallet connection
    initializeWallet();

    // Set up navigation
    initializeNavigation();

    // Load initial data
    loadWalletData();
    loadMarketplaceData();
    loadMetaverseData();
    loadStakingData();

    // Start periodic updates
    startPeriodicUpdates();

    console.log('✅ LL TOKEN OFFLINE - Ready!');
    showToast('LL TOKEN Wallet loaded successfully', 'success');
});

// Wallet Initialization
function initializeWallet() {
    // Generate mock wallet ID for demo
    const walletId = generateMockWalletId();
    LLToken.walletId = walletId;
    LLToken.walletConnected = true;

    // Update wallet ID in UI
    const walletIdElement = document.getElementById('walletId');
    if (walletIdElement) {
        walletIdElement.textContent = walletId;
    }

    console.log('🔐 Wallet initialized:', walletId);
}

function generateMockWalletId() {
    const chars = '0123456789ABCDEF';
    let result = 'WALLET_';
    for (let i = 0; i < 16; i++) {
        result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
}

// Navigation System
function initializeNavigation() {
    const tabs = ['walletTab', 'marketplaceTab', 'metaverseTab', 'stakingTab'];
    const sections = ['walletSection', 'marketplaceSection', 'metaverseSection', 'stakingSection'];

    tabs.forEach((tabId, index) => {
        const tabElement = document.getElementById(tabId);
        if (tabElement) {
            tabElement.addEventListener('click', () => {
                switchTab(tabId.replace('Tab', ''), sections[index]);
            });
        }
    });
}

function switchTab(tabName, sectionId) {
    // Update current tab
    LLToken.currentTab = tabName;

    // Update navigation buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.getElementById(tabName + 'Tab').classList.add('active');

    // Show/hide sections
    document.querySelectorAll('.tab-content').forEach(section => {
        section.classList.remove('active');
        section.classList.add('hidden');
    });

    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.classList.remove('hidden');
        targetSection.classList.add('active');
    }

    console.log('📍 Switched to tab:', tabName);
}

// Load Wallet Data
function loadWalletData() {
    // Generate mock token holdings
    const mockHoldings = generateMockTokenHoldings();
    LLToken.tokenHoldings = mockHoldings;

    // Calculate total balance
    let totalBalance = 0;
    Object.values(mockHoldings).forEach(holding => {
        totalBalance += holding.balance;
    });
    LLToken.totalBalance = totalBalance;

    // Update UI
    updateWalletUI();
    generateMockTransactions();
}

function generateMockTokenHoldings() {
    const holdings = {};
    const tokenTypes = Object.keys(LLToken.tokenSpecs);

    tokenTypes.forEach(tokenType => {
        holdings[tokenType] = {
            balance: Math.floor(Math.random() * 50000) + 1000,
            usdValue: (Math.random() * 1000 + 50).toFixed(2),
            change24h: (Math.random() * 20 - 10).toFixed(2)
        };
    });

    return holdings;
}

function updateWalletUI() {
    // Update total balance
    const totalBalanceElement = document.getElementById('totalBalance');
    if (totalBalanceElement) {
        totalBalanceElement.textContent = LLToken.totalBalance.toLocaleString();
    }

    // Update USD value (mock calculation)
    const usdValueElement = document.getElementById('usdValue');
    if (usdValueElement) {
        const mockUsdValue = (LLToken.totalBalance * 0.05).toFixed(2);
        usdValueElement.textContent = `(≈ $${mockUsdValue})`;
    }

    // Update token holdings
    updateTokenHoldingsUI();
}

function updateTokenHoldingsUI() {
    const holdingsContainer = document.getElementById('tokenHoldings');
    if (!holdingsContainer) return;

    holdingsContainer.innerHTML = '';

    Object.entries(LLToken.tokenHoldings).forEach(([tokenType, holding]) => {
        const tokenSpec = LLToken.tokenSpecs[tokenType];
        const changeClass = parseFloat(holding.change24h) >= 0 ? 'text-green-400' : 'text-red-400';
        const changeIcon = parseFloat(holding.change24h) >= 0 ? '↑' : '↓';

        const tokenCard = document.createElement('div');
        tokenCard.className = 'token-card';
        tokenCard.innerHTML = `
            <div class="flex justify-between items-center">
                <div class="flex items-center space-x-3">
                    <div class="w-10 h-10 rounded-full flex items-center justify-center" style="background-color: ${tokenSpec.color}20; border: 1px solid ${tokenSpec.color}40;">
                        <span class="text-sm font-bold" style="color: ${tokenSpec.color};">${tokenSpec.symbol.split('-')[1][0]}</span>
                    </div>
                    <div>
                        <p class="font-medium">${tokenSpec.name}</p>
                        <p class="text-sm text-gray-400">${tokenSpec.symbol}</p>
                    </div>
                </div>
                <div class="text-right">
                    <p class="font-semibold">${holding.balance.toLocaleString()}</p>
                    <p class="text-sm text-gray-400">$${holding.usdValue}</p>
                    <p class="text-xs ${changeClass}">${changeIcon} ${Math.abs(holding.change24h)}%</p>
                </div>
            </div>
            <div class="mt-3 pt-3 border-t border-gray-600">
                <p class="text-xs text-gray-400">Primary Utilities:</p>
                <p class="text-xs text-gray-300">${tokenSpec.utilities.slice(0, 2).join(', ')}</p>
            </div>
        `;

        holdingsContainer.appendChild(tokenCard);
    });
}

function generateMockTransactions() {
    const transactions = [];
    const transactionTypes = ['FL Reward', 'Avatar Purchase', 'Land Trade', 'Staking Reward', 'Asset Creation'];
    const tokenTypes = Object.keys(LLToken.tokenSpecs);

    for (let i = 0; i < 10; i++) {
        const tokenType = tokenTypes[Math.floor(Math.random() * tokenTypes.length)];
        const amount = Math.floor(Math.random() * 1000) + 10;
        const type = transactionTypes[Math.floor(Math.random() * transactionTypes.length)];
        const isIncoming = Math.random() > 0.5;

        transactions.push({
            id: `TX-${Date.now()}-${i}`,
            type: type,
            tokenType: tokenType,
            amount: amount,
            direction: isIncoming ? 'in' : 'out',
            timestamp: new Date(Date.now() - Math.random() * 86400000 * 7), // Last 7 days
            status: 'confirmed'
        });
    }

    LLToken.transactions = transactions.sort((a, b) => b.timestamp - a.timestamp);
    updateTransactionsUI();
}

function updateTransactionsUI() {
    const transactionsContainer = document.getElementById('recentTransactions');
    if (!transactionsContainer) return;

    transactionsContainer.innerHTML = '';

    LLToken.transactions.slice(0, 5).forEach(transaction => {
        const tokenSpec = LLToken.tokenSpecs[transaction.tokenType];
        const directionIcon = transaction.direction === 'in' ? '📥' : '📤';
        const directionColor = transaction.direction === 'in' ? 'text-green-400' : 'text-red-400';
        const amountSign = transaction.direction === 'in' ? '+' : '-';

        const transactionItem = document.createElement('div');
        transactionItem.className = 'transaction-item';
        transactionItem.innerHTML = `
            <div class="flex items-center space-x-3">
                <div class="text-2xl">${directionIcon}</div>
                <div>
                    <p class="font-medium">${transaction.type}</p>
                    <p class="text-sm text-gray-400">${transaction.timestamp.toLocaleDateString()}</p>
                </div>
            </div>
            <div class="text-right">
                <p class="font-semibold ${directionColor}">${amountSign}${transaction.amount.toLocaleString()}</p>
                <p class="text-sm text-gray-400">${tokenSpec.symbol}</p>
                <p class="text-xs text-green-400">Confirmed</p>
            </div>
        `;

        transactionsContainer.appendChild(transactionItem);
    });
}

// Load Marketplace Data
function loadMarketplaceData() {
    generateMockMarketData();
    generateMockOrderBook();
    generateMockNFTs();
}

function generateMockMarketData() {
    // Update market stats
    const stats = {
        totalVolume: (Math.random() * 50 + 10).toFixed(1) + 'M',
        activeListings: Math.floor(Math.random() * 2000) + 500,
        totalUsers: Math.floor(Math.random() * 10000) + 5000
    };

    document.getElementById('totalVolume').textContent = stats.totalVolume;
    document.getElementById('activeListings').textContent = stats.activeListings.toLocaleString();
    document.getElementById('totalUsers').textContent = stats.totalUsers.toLocaleString();
}

function generateMockOrderBook() {
    // Generate sell orders
    const sellOrdersContainer = document.getElementById('sellOrders');
    const buyOrdersContainer = document.getElementById('buyOrders');

    if (sellOrdersContainer) {
        sellOrdersContainer.innerHTML = '';
        for (let i = 0; i < 5; i++) {
            const price = (1.05 + i * 0.02).toFixed(3);
            const amount = Math.floor(Math.random() * 10000) + 1000;

            const orderItem = document.createElement('div');
            orderItem.className = 'flex justify-between text-sm';
            orderItem.innerHTML = `
                <span class="text-red-400">${price}</span>
                <span class="text-gray-300">${amount.toLocaleString()}</span>
            `;
            sellOrdersContainer.appendChild(orderItem);
        }
    }

    if (buyOrdersContainer) {
        buyOrdersContainer.innerHTML = '';
        for (let i = 0; i < 5; i++) {
            const price = (0.95 - i * 0.02).toFixed(3);
            const amount = Math.floor(Math.random() * 10000) + 1000;

            const orderItem = document.createElement('div');
            orderItem.className = 'flex justify-between text-sm';
            orderItem.innerHTML = `
                <span class="text-green-400">${price}</span>
                <span class="text-gray-300">${amount.toLocaleString()}</span>
            `;
            buyOrdersContainer.appendChild(orderItem);
        }
    }
}

function generateMockNFTs() {
    const nftsContainer = document.getElementById('featuredNFTs');
    if (!nftsContainer) return;

    nftsContainer.innerHTML = '';

    const nftTypes = [
        { name: 'Quantum Avatar', emoji: '🧑‍🎤', price: '2,500' },
        { name: 'Virtual Land', emoji: '🏝️', price: '15,000' },
        { name: 'Cosmic Sword', emoji: '⚔️', price: '850' },
        { name: 'Wisdom Orb', emoji: '🔮', price: '3,200' }
    ];

    nftTypes.forEach(nft => {
        const nftCard = document.createElement('div');
        nftCard.className = 'bg-white/5 border border-purple-500/20 rounded-lg p-4 hover:bg-white/8 transition-all cursor-pointer';
        nftCard.innerHTML = `
            <div class="text-4xl text-center mb-3">${nft.emoji}</div>
            <h4 class="font-medium text-center mb-2">${nft.name}</h4>
            <p class="text-sm text-gray-400 text-center mb-3">Starting at</p>
            <p class="text-lg font-semibold text-center text-purple-400">${nft.price} LLT</p>
            <button class="w-full mt-3 bg-purple-500/20 border border-purple-500/30 py-2 px-4 rounded-lg hover:bg-purple-500/30 transition-all text-sm">
                View Details
            </button>
        `;
        nftsContainer.appendChild(nftCard);
    });
}

// Load Metaverse Data
function loadMetaverseData() {
    generateVirtualWorlds();
    generateVirtualAssets();
}

function generateVirtualWorlds() {
    const worldsContainer = document.getElementById('virtualWorlds');
    if (!worldsContainer) return;

    const worlds = [
        { name: 'VRChat Universe', status: 'Connected', users: '1,247', emoji: '🌐' },
        { name: 'Horizon Worlds', status: 'Available', users: '892', emoji: '🌅' },
        { name: 'Decentraland', status: 'Connected', users: '634', emoji: '🏙️' },
        { name: 'The Sandbox', status: 'Available', users: '1,103', emoji: '🏖️' }
    ];

    worldsContainer.innerHTML = '';

    worlds.forEach(world => {
        const statusColor = world.status === 'Connected' ? 'text-green-400' : 'text-yellow-400';
        const statusDot = world.status === 'Connected' ? 'bg-green-500' : 'bg-yellow-500';

        const worldCard = document.createElement('div');
        worldCard.className = 'flex items-center justify-between p-4 bg-white/5 border border-purple-500/20 rounded-lg hover:bg-white/8 transition-all';
        worldCard.innerHTML = `
            <div class="flex items-center space-x-3">
                <div class="text-2xl">${world.emoji}</div>
                <div>
                    <p class="font-medium">${world.name}</p>
                    <p class="text-sm text-gray-400">${world.users} active users</p>
                </div>
            </div>
            <div class="flex items-center space-x-2">
                <div class="w-2 h-2 ${statusDot} rounded-full"></div>
                <span class="text-sm ${statusColor}">${world.status}</span>
            </div>
        `;
        worldsContainer.appendChild(worldCard);
    });
}

function generateVirtualAssets() {
    const assetsContainer = document.getElementById('virtualAssets');
    if (!assetsContainer) return;

    const assets = [
        { name: 'Quantum Blade', type: 'Weapon', rarity: 'Legendary', emoji: '⚔️' },
        { name: 'Wisdom Crown', type: 'Accessory', rarity: 'Epic', emoji: '👑' },
        { name: 'Portal Stone', type: 'Tool', rarity: 'Rare', emoji: '🌀' },
        { name: 'Energy Crystal', type: 'Resource', rarity: 'Common', emoji: '💎' }
    ];

    assetsContainer.innerHTML = '';

    assets.forEach(asset => {
        const rarityColors = {
            'Legendary': 'text-yellow-400 border-yellow-400/30',
            'Epic': 'text-purple-400 border-purple-400/30',
            'Rare': 'text-blue-400 border-blue-400/30',
            'Common': 'text-gray-400 border-gray-400/30'
        };

        const assetCard = document.createElement('div');
        assetCard.className = `bg-white/5 border rounded-lg p-4 hover:bg-white/8 transition-all cursor-pointer ${rarityColors[asset.rarity]}`;
        assetCard.innerHTML = `
            <div class="text-3xl text-center mb-3">${asset.emoji}</div>
            <h4 class="font-medium text-center mb-1">${asset.name}</h4>
            <p class="text-xs text-gray-400 text-center mb-2">${asset.type}</p>
            <p class="text-xs text-center ${rarityColors[asset.rarity].split(' ')[0]}">${asset.rarity}</p>
        `;
        assetsContainer.appendChild(assetCard);
    });
}

// Load Staking Data
function loadStakingData() {
    generateStakingPools();
    updateStakingOverview();
}

function updateStakingOverview() {
    const totalStaked = Math.floor(Math.random() * 100000) + 50000;
    const pendingRewards = Math.floor(Math.random() * 5000) + 1000;

    document.getElementById('totalStaked').textContent = `${totalStaked.toLocaleString()} LLT`;
    document.getElementById('pendingRewards').textContent = `${pendingRewards.toLocaleString()} LLT`;
}

function generateStakingPools() {
    const poolsContainer = document.getElementById('stakingPools');
    if (!poolsContainer) return;

    const pools = [
        { name: 'LLT-STAKE Pool', apy: '8.5%', tvl: '2.4M', minStake: '1,000' },
        { name: 'LLT-GOV Pool', apy: '12.3%', tvl: '856K', minStake: '5,000' },
        { name: 'LLT-COMPUTE Pool', apy: '6.8%', tvl: '1.8M', minStake: '500' }
    ];

    poolsContainer.innerHTML = '';

    pools.forEach(pool => {
        const poolCard = document.createElement('div');
        poolCard.className = 'bg-white/5 border border-purple-500/20 rounded-lg p-4 hover:bg-white/8 transition-all';
        poolCard.innerHTML = `
            <div class="flex justify-between items-start mb-3">
                <h4 class="font-medium">${pool.name}</h4>
                <span class="text-sm bg-green-500/20 text-green-400 px-2 py-1 rounded">${pool.apy} APY</span>
            </div>
            <div class="grid grid-cols-2 gap-4 text-sm">
                <div>
                    <p class="text-gray-400">TVL</p>
                    <p class="font-medium">${pool.tvl} LLT</p>
                </div>
                <div>
                    <p class="text-gray-400">Min Stake</p>
                    <p class="font-medium">${pool.minStake} LLT</p>
                </div>
            </div>
            <button class="w-full mt-3 bg-blue-500/20 border border-blue-500/30 py-2 px-4 rounded-lg hover:bg-blue-500/30 transition-all text-sm">
                Stake Now
            </button>
        `;
        poolsContainer.appendChild(poolCard);
    });
}

// Periodic Updates
function startPeriodicUpdates() {
    setInterval(() => {
        // Update market data
        if (LLToken.currentTab === 'marketplace') {
            generateMockMarketData();
            generateMockOrderBook();
        }

        // Update staking rewards
        if (LLToken.currentTab === 'staking') {
            updateStakingOverview();
        }

        // Simulate new transactions occasionally
        if (Math.random() < 0.1) { // 10% chance every update
            addNewMockTransaction();
        }
    }, 30000); // Update every 30 seconds
}

function addNewMockTransaction() {
    const tokenTypes = Object.keys(LLToken.tokenSpecs);
    const transactionTypes = ['FL Reward', 'Avatar Purchase', 'Staking Reward'];

    const newTransaction = {
        id: `TX-${Date.now()}`,
        type: transactionTypes[Math.floor(Math.random() * transactionTypes.length)],
        tokenType: tokenTypes[Math.floor(Math.random() * tokenTypes.length)],
        amount: Math.floor(Math.random() * 500) + 10,
        direction: 'in', // New transactions are usually incoming rewards
        timestamp: new Date(),
        status: 'confirmed'
    };

    LLToken.transactions.unshift(newTransaction);
    LLToken.transactions = LLToken.transactions.slice(0, 50); // Keep only latest 50

    if (LLToken.currentTab === 'wallet') {
        updateTransactionsUI();
    }

    showToast(`New ${newTransaction.type}: +${newTransaction.amount} ${LLToken.tokenSpecs[newTransaction.tokenType].symbol}`, 'success');
}

// Utility Functions
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <div class="flex items-center space-x-2">
            <div class="text-lg">${type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️'}</div>
            <span>${message}</span>
        </div>
    `;

    document.body.appendChild(toast);

    // Show toast
    setTimeout(() => {
        toast.classList.add('show');
    }, 100);

    // Hide toast after 3 seconds
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 300);
    }, 3000);
}

function showLoading(message = 'Processing...') {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.remove('hidden');
    }
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.add('hidden');
    }
}

// Export for use in other modules
window.LLTokenApp = {
    showToast,
    showLoading,
    hideLoading,
    switchTab
};