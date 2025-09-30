/**
 * LL TOKEN OFFLINE - Marketplace Module
 * Token trading and NFT marketplace functionality
 */

const MarketplaceManager = {
    // Current market data
    marketData: {
        pairs: {},
        orderBook: {},
        recentTrades: [],
        featuredNFTs: [],
        collections: {}
    },

    // Trading configuration
    config: {
        minTradeAmount: 0.000001,
        maxSlippage: 0.05, // 5%
        tradingFee: 0.002, // 0.2%
        makerFee: 0.001,   // 0.1%
        takerFee: 0.002    // 0.2%
    },

    // Initialize marketplace
    initialize() {
        this.loadMarketPairs();
        this.generateOrderBook();
        this.loadFeaturedNFTs();
        this.setupTradingInterface();
        this.startMarketUpdates();

        console.log('🏪 Marketplace initialized');
    },

    // Load trading pairs
    loadMarketPairs() {
        const tokens = Object.keys(LLToken.tokenSpecs);

        // Create trading pairs
        tokens.forEach(baseToken => {
            tokens.forEach(quoteToken => {
                if (baseToken !== quoteToken) {
                    const pairKey = `${baseToken}/${quoteToken}`;
                    this.marketData.pairs[pairKey] = {
                        base: baseToken,
                        quote: quoteToken,
                        price: this.calculatePairPrice(baseToken, quoteToken),
                        change24h: (Math.random() * 20 - 10).toFixed(2), // -10% to +10%
                        volume24h: Math.floor(Math.random() * 1000000) + 10000,
                        high24h: 0,
                        low24h: 0,
                        lastTrade: new Date()
                    };
                }
            });
        });

        // Set high/low values based on current price
        Object.values(this.marketData.pairs).forEach(pair => {
            const variance = 0.1; // 10% variance
            pair.high24h = pair.price * (1 + variance);
            pair.low24h = pair.price * (1 - variance);
        });
    },

    // Calculate pair price
    calculatePairPrice(baseToken, quoteToken) {
        const baseValue = this.getTokenValue(baseToken);
        const quoteValue = this.getTokenValue(quoteToken);
        return (baseValue / quoteValue) * (0.9 + Math.random() * 0.2); // Add 10% variance
    },

    // Get token relative value
    getTokenValue(tokenType) {
        const values = {
            'LLT-GOV': 10.0,     // Most valuable - governance
            'LLT-LAND': 8.5,     // Scarce land tokens
            'LLT-STAKE': 6.2,    // Staking utility
            'LLT-COMPUTE': 4.8,  // Computing power
            'LLT-AVATAR': 3.5,   // Base metaverse token
            'LLT-ASSET': 3.2,    // Asset tokens
            'LLT-CREATE': 2.8,   // Creator tokens
            'LLT-DATA': 2.5,     // Data tokens
            'LLT-REP': 1.8,      // Reputation (soul-bound, less tradeable)
            'LLT-EXP': 1.0       // Experience tokens (most common)
        };
        return values[tokenType] || 1.0;
    },

    // Generate order book
    generateOrderBook() {
        const activePairs = Object.keys(this.marketData.pairs).slice(0, 10); // Top 10 pairs

        activePairs.forEach(pairKey => {
            const pair = this.marketData.pairs[pairKey];
            const midPrice = pair.price;

            // Generate buy orders (bids)
            const bids = [];
            for (let i = 0; i < 10; i++) {
                const priceStep = midPrice * 0.01 * (i + 1); // 1% steps
                bids.push({
                    price: midPrice - priceStep,
                    amount: Math.floor(Math.random() * 10000) + 1000,
                    total: 0,
                    side: 'buy'
                });
            }

            // Generate sell orders (asks)
            const asks = [];
            for (let i = 0; i < 10; i++) {
                const priceStep = midPrice * 0.01 * (i + 1); // 1% steps
                asks.push({
                    price: midPrice + priceStep,
                    amount: Math.floor(Math.random() * 10000) + 1000,
                    total: 0,
                    side: 'sell'
                });
            }

            // Calculate running totals
            let bidTotal = 0;
            bids.forEach(bid => {
                bidTotal += bid.amount;
                bid.total = bidTotal;
            });

            let askTotal = 0;
            asks.reverse().forEach(ask => {
                askTotal += ask.amount;
                ask.total = askTotal;
            });
            asks.reverse();

            this.marketData.orderBook[pairKey] = {
                bids: bids,
                asks: asks,
                spread: asks[0].price - bids[0].price,
                spreadPercent: ((asks[0].price - bids[0].price) / midPrice * 100).toFixed(2)
            };
        });
    },

    // Execute trade
    async executeTrade(side, pair, amount, price = null) {
        try {
            LLTokenApp.showLoading('Processing trade...');

            const [baseToken, quoteToken] = pair.split('/');
            const pairData = this.marketData.pairs[pair];

            if (!pairData) {
                throw new Error('Trading pair not found');
            }

            // Use market price if no price specified
            const tradePrice = price || pairData.price;
            const totalCost = amount * tradePrice;
            const fee = totalCost * this.config.tradingFee;

            // Check balances
            if (side === 'buy') {
                const quoteBalance = LLToken.tokenHoldings[quoteToken]?.balance || 0;
                if (quoteBalance < totalCost + fee) {
                    throw new Error(`Insufficient ${quoteToken} balance`);
                }
            } else {
                const baseBalance = LLToken.tokenHoldings[baseToken]?.balance || 0;
                if (baseBalance < amount) {
                    throw new Error(`Insufficient ${baseToken} balance`);
                }
            }

            // Simulate trade execution delay
            await new Promise(resolve => setTimeout(resolve, 2000));

            // Create trade record
            const trade = {
                id: `TRADE-${Date.now()}-${Math.random().toString(36).substr(2, 6).toUpperCase()}`,
                pair: pair,
                side: side,
                amount: amount,
                price: tradePrice,
                total: totalCost,
                fee: fee,
                timestamp: new Date(),
                status: 'filled'
            };

            // Update balances
            if (side === 'buy') {
                // Buying base token with quote token
                LLToken.tokenHoldings[quoteToken].balance -= totalCost + fee;
                if (!LLToken.tokenHoldings[baseToken]) {
                    LLToken.tokenHoldings[baseToken] = { balance: 0, usdValue: '0', change24h: '0' };
                }
                LLToken.tokenHoldings[baseToken].balance += amount;
            } else {
                // Selling base token for quote token
                LLToken.tokenHoldings[baseToken].balance -= amount;
                if (!LLToken.tokenHoldings[quoteToken]) {
                    LLToken.tokenHoldings[quoteToken] = { balance: 0, usdValue: '0', change24h: '0' };
                }
                LLToken.tokenHoldings[quoteToken].balance += totalCost - fee;
            }

            // Add to transaction history
            LLToken.transactions.unshift({
                id: trade.id,
                type: 'Trade',
                tokenType: baseToken,
                amount: side === 'buy' ? amount : -amount,
                direction: side === 'buy' ? 'in' : 'out',
                timestamp: trade.timestamp,
                status: 'confirmed',
                metadata: {
                    pair: pair,
                    side: side,
                    price: tradePrice,
                    fee: fee
                }
            });

            // Update market data
            this.updateMarketData(pair, tradePrice, amount);

            // Update UI
            if (typeof WalletManager !== 'undefined' && WalletManager.updateWalletUI) {
                WalletManager.updateWalletUI();
            }

            LLTokenApp.hideLoading();
            LLTokenApp.showToast(`Trade executed: ${side} ${amount} ${baseToken} at ${tradePrice.toFixed(6)}`, 'success');

            return trade;

        } catch (error) {
            LLTokenApp.hideLoading();
            LLTokenApp.showToast(`Trade failed: ${error.message}`, 'error');
            throw error;
        }
    },

    // Update market data after trade
    updateMarketData(pair, price, volume) {
        const pairData = this.marketData.pairs[pair];
        if (pairData) {
            pairData.price = price;
            pairData.volume24h += volume;
            pairData.lastTrade = new Date();

            // Update high/low
            if (price > pairData.high24h) pairData.high24h = price;
            if (price < pairData.low24h) pairData.low24h = price;

            // Add to recent trades
            this.marketData.recentTrades.unshift({
                pair: pair,
                price: price,
                amount: volume,
                timestamp: new Date(),
                side: Math.random() > 0.5 ? 'buy' : 'sell'
            });

            // Keep only last 50 trades
            this.marketData.recentTrades = this.marketData.recentTrades.slice(0, 50);
        }
    },

    // Load featured NFTs
    loadFeaturedNFTs() {
        const nftCollections = [
            {
                name: 'Quantum Avatars',
                description: 'Unique quantum-enhanced avatars for the metaverse',
                items: [
                    { id: 1, name: 'Quantum Warrior', price: 2500, rarity: 'Legendary', image: '🛡️' },
                    { id: 2, name: 'Cyber Mage', price: 1800, rarity: 'Epic', image: '🧙‍♂️' },
                    { id: 3, name: 'Data Guardian', price: 1200, rarity: 'Rare', image: '👾' }
                ]
            },
            {
                name: 'Virtual Lands',
                description: 'Prime real estate in the LL TOKEN metaverse',
                items: [
                    { id: 4, name: 'Quantum City Plot', price: 15000, rarity: 'Legendary', image: '🏙️' },
                    { id: 5, name: 'Crystal Cave', price: 8500, rarity: 'Epic', image: '💎' },
                    { id: 6, name: 'Floating Island', price: 5200, rarity: 'Rare', image: '🏝️' }
                ]
            },
            {
                name: 'Cosmic Items',
                description: 'Powerful items for your adventures',
                items: [
                    { id: 7, name: 'Infinity Sword', price: 3200, rarity: 'Legendary', image: '⚔️' },
                    { id: 8, name: 'Wisdom Orb', price: 1500, rarity: 'Epic', image: '🔮' },
                    { id: 9, name: 'Speed Boots', price: 800, rarity: 'Rare', image: '👢' }
                ]
            }
        ];

        this.marketData.featuredNFTs = nftCollections.flatMap(collection =>
            collection.items.map(item => ({
                ...item,
                collection: collection.name,
                description: collection.description
            }))
        );
    },

    // Buy NFT
    async buyNFT(nftId, paymentToken = 'LLT-AVATAR') {
        try {
            LLTokenApp.showLoading('Processing NFT purchase...');

            const nft = this.marketData.featuredNFTs.find(n => n.id === nftId);
            if (!nft) {
                throw new Error('NFT not found');
            }

            // Check balance
            const balance = LLToken.tokenHoldings[paymentToken]?.balance || 0;
            if (balance < nft.price) {
                throw new Error(`Insufficient ${paymentToken} balance`);
            }

            // Simulate purchase process
            await new Promise(resolve => setTimeout(resolve, 3000));

            // Create purchase transaction
            const purchase = {
                id: `NFT-${Date.now()}`,
                type: 'NFT Purchase',
                nftId: nft.id,
                nftName: nft.name,
                collection: nft.collection,
                price: nft.price,
                paymentToken: paymentToken,
                timestamp: new Date(),
                status: 'confirmed'
            };

            // Update balance
            LLToken.tokenHoldings[paymentToken].balance -= nft.price;

            // Add to transactions
            LLToken.transactions.unshift({
                id: purchase.id,
                type: purchase.type,
                tokenType: paymentToken,
                amount: -nft.price,
                direction: 'out',
                timestamp: purchase.timestamp,
                status: 'confirmed',
                metadata: {
                    nftName: nft.name,
                    collection: nft.collection,
                    rarity: nft.rarity
                }
            });

            // Update UI
            if (typeof WalletManager !== 'undefined' && WalletManager.updateWalletUI) {
                WalletManager.updateWalletUI();
            }

            LLTokenApp.hideLoading();
            LLTokenApp.showToast(`Successfully purchased ${nft.name} for ${nft.price} ${paymentToken}`, 'success');

            return purchase;

        } catch (error) {
            LLTokenApp.hideLoading();
            LLTokenApp.showToast(`NFT purchase failed: ${error.message}`, 'error');
            throw error;
        }
    },

    // Get market statistics
    getMarketStats() {
        const pairs = Object.values(this.marketData.pairs);

        return {
            totalPairs: pairs.length,
            totalVolume24h: pairs.reduce((sum, pair) => sum + pair.volume24h, 0),
            topGainer: pairs.reduce((max, pair) =>
                parseFloat(pair.change24h) > parseFloat(max.change24h) ? pair : max
            ),
            topLoser: pairs.reduce((min, pair) =>
                parseFloat(pair.change24h) < parseFloat(min.change24h) ? pair : min
            ),
            averageChange: pairs.reduce((sum, pair) => sum + parseFloat(pair.change24h), 0) / pairs.length
        };
    },

    // Setup trading interface
    setupTradingInterface() {
        // Update trading pair selector
        const tradingPairSelect = document.getElementById('tradingPair');
        if (tradingPairSelect) {
            tradingPairSelect.innerHTML = '';

            // Add popular pairs
            const popularPairs = [
                'LLT-AVATAR/LLT-COMPUTE',
                'LLT-LAND/LLT-GOV',
                'LLT-ASSET/LLT-CREATE',
                'LLT-STAKE/LLT-GOV',
                'LLT-DATA/LLT-COMPUTE'
            ];

            popularPairs.forEach(pair => {
                if (this.marketData.pairs[pair]) {
                    const option = document.createElement('option');
                    option.value = pair;
                    option.textContent = pair.replace('LLT-', '');
                    tradingPairSelect.appendChild(option);
                }
            });

            // Add change listener for trading pair
            tradingPairSelect.addEventListener('change', (e) => {
                this.updateTradingInterface(e.target.value);
            });

            // Initialize with first pair
            if (popularPairs.length > 0) {
                this.updateTradingInterface(popularPairs[0]);
            }
        }

        // Setup trading form
        this.setupTradingForm();
    },

    // Update trading interface for selected pair
    updateTradingInterface(pairKey) {
        const pair = this.marketData.pairs[pairKey];
        if (!pair) return;

        const orderBook = this.marketData.orderBook[pairKey];
        if (orderBook) {
            this.updateOrderBookUI(orderBook);
        }

        // Update pair info (could add price display, charts, etc.)
        console.log(`Updated trading interface for ${pairKey}:`, pair);
    },

    // Update order book UI
    updateOrderBookUI(orderBook) {
        const sellOrdersContainer = document.getElementById('sellOrders');
        const buyOrdersContainer = document.getElementById('buyOrders');

        if (sellOrdersContainer) {
            sellOrdersContainer.innerHTML = '';
            orderBook.asks.slice(0, 5).forEach(ask => {
                const orderItem = document.createElement('div');
                orderItem.className = 'flex justify-between text-sm cursor-pointer hover:bg-white/5 p-1 rounded';
                orderItem.innerHTML = `
                    <span class="text-red-400">${ask.price.toFixed(6)}</span>
                    <span class="text-gray-300">${ask.amount.toLocaleString()}</span>
                `;
                orderItem.addEventListener('click', () => this.fillOrderFromBook('sell', ask.price, ask.amount));
                sellOrdersContainer.appendChild(orderItem);
            });
        }

        if (buyOrdersContainer) {
            buyOrdersContainer.innerHTML = '';
            orderBook.bids.slice(0, 5).forEach(bid => {
                const orderItem = document.createElement('div');
                orderItem.className = 'flex justify-between text-sm cursor-pointer hover:bg-white/5 p-1 rounded';
                orderItem.innerHTML = `
                    <span class="text-green-400">${bid.price.toFixed(6)}</span>
                    <span class="text-gray-300">${bid.amount.toLocaleString()}</span>
                `;
                orderItem.addEventListener('click', () => this.fillOrderFromBook('buy', bid.price, bid.amount));
                buyOrdersContainer.appendChild(orderItem);
            });
        }
    },

    // Fill order from order book click
    fillOrderFromBook(side, price, amount) {
        // This would populate the trading form with the selected order
        console.log(`Selected ${side} order: ${amount} at ${price}`);
        LLTokenApp.showToast(`Selected ${side} order: ${amount.toLocaleString()} at ${price.toFixed(6)}`, 'info');
    },

    // Setup trading form
    setupTradingForm() {
        // Add trading form functionality
        const tradeForm = document.querySelector('.grid.grid-cols-2.gap-4');
        if (tradeForm) {
            // Add execute trade button listener
            const executeButton = tradeForm.parentElement.querySelector('button');
            if (executeButton) {
                executeButton.addEventListener('click', this.handleTradeExecution.bind(this));
            }
        }
    },

    // Handle trade execution
    async handleTradeExecution() {
        try {
            const tradingPairSelect = document.getElementById('tradingPair');
            const payInputs = document.querySelectorAll('.grid.grid-cols-2.gap-4 input');

            if (!tradingPairSelect || payInputs.length < 2) {
                throw new Error('Trading interface not properly initialized');
            }

            const pair = tradingPairSelect.value;
            const payAmount = parseFloat(payInputs[0].value);
            const receiveAmount = parseFloat(payInputs[1].value);

            if (!pair || !payAmount || payAmount <= 0) {
                throw new Error('Please enter a valid amount to trade');
            }

            // Determine trade side based on input
            const [baseToken, quoteToken] = pair.split('/');
            const side = 'buy'; // Simplified - assume buying base with quote

            await this.executeTrade(side, pair, receiveAmount || payAmount);

            // Clear form
            payInputs.forEach(input => input.value = '');

        } catch (error) {
            console.error('Trade execution error:', error);
        }
    },

    // Start periodic market updates
    startMarketUpdates() {
        setInterval(() => {
            this.updateMarketPrices();
            this.generateRecentTrades();
        }, 10000); // Update every 10 seconds
    },

    // Update market prices (simulate price movements)
    updateMarketPrices() {
        Object.values(this.marketData.pairs).forEach(pair => {
            // Simulate price movement (±2%)
            const change = (Math.random() - 0.5) * 0.04;
            pair.price *= (1 + change);

            // Update 24h change
            const oldChange = parseFloat(pair.change24h);
            pair.change24h = (oldChange + change * 100).toFixed(2);

            // Update high/low if needed
            if (pair.price > pair.high24h) pair.high24h = pair.price;
            if (pair.price < pair.low24h) pair.low24h = pair.price;
        });

        // Update order books
        this.generateOrderBook();
    },

    // Generate recent trades
    generateRecentTrades() {
        const activePairs = Object.keys(this.marketData.pairs).slice(0, 5);

        if (Math.random() < 0.3) { // 30% chance of new trade
            const randomPair = activePairs[Math.floor(Math.random() * activePairs.length)];
            const pairData = this.marketData.pairs[randomPair];

            if (pairData) {
                const trade = {
                    pair: randomPair,
                    price: pairData.price * (0.98 + Math.random() * 0.04), // ±2% from current price
                    amount: Math.floor(Math.random() * 5000) + 100,
                    timestamp: new Date(),
                    side: Math.random() > 0.5 ? 'buy' : 'sell'
                };

                this.marketData.recentTrades.unshift(trade);
                this.marketData.recentTrades = this.marketData.recentTrades.slice(0, 50);
            }
        }
    }
};

// NFT Manager
const NFTManager = {
    // User's NFT collection
    userNFTs: [],

    // Initialize NFT system
    initialize() {
        this.loadUserNFTs();
        this.setupNFTInterface();
    },

    // Load user's NFT collection
    loadUserNFTs() {
        // Generate mock NFT collection based on transaction history
        const nftPurchases = LLToken.transactions.filter(tx => tx.type === 'NFT Purchase');

        this.userNFTs = nftPurchases.map(purchase => ({
            id: purchase.id,
            name: purchase.metadata?.nftName || 'Unknown NFT',
            collection: purchase.metadata?.collection || 'Unknown Collection',
            rarity: purchase.metadata?.rarity || 'Common',
            acquiredDate: purchase.timestamp,
            estimatedValue: Math.floor(Math.random() * 5000) + 500
        }));
    },

    // Setup NFT interface
    setupNFTInterface() {
        // Add click listeners to featured NFTs
        document.addEventListener('click', (e) => {
            if (e.target.closest('[data-nft-id]')) {
                const nftElement = e.target.closest('[data-nft-id]');
                const nftId = parseInt(nftElement.dataset.nftId);
                this.showNFTDetails(nftId);
            }
        });
    },

    // Show NFT details modal
    showNFTDetails(nftId) {
        const nft = MarketplaceManager.marketData.featuredNFTs.find(n => n.id === nftId);
        if (!nft) return;

        const modal = this.createNFTModal(nft);
        document.body.appendChild(modal);
    },

    // Create NFT modal
    createNFTModal(nft) {
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50';
        modal.innerHTML = `
            <div class="bg-white/10 backdrop-blur-lg border border-purple-500/30 rounded-xl p-6 max-w-lg w-full mx-4">
                <div class="flex justify-between items-center mb-6">
                    <h3 class="text-xl font-semibold">${nft.name}</h3>
                    <button onclick="this.closest('.fixed').remove()" class="text-gray-400 hover:text-white text-2xl">&times;</button>
                </div>

                <div class="text-center mb-6">
                    <div class="text-6xl mb-4">${nft.image}</div>
                    <h4 class="text-lg font-medium mb-2">${nft.name}</h4>
                    <p class="text-sm text-gray-400 mb-2">${nft.collection}</p>
                    <span class="inline-block px-3 py-1 bg-purple-500/20 text-purple-300 rounded-full text-sm">
                        ${nft.rarity}
                    </span>
                </div>

                <div class="space-y-4 mb-6">
                    <div class="flex justify-between">
                        <span class="text-gray-400">Price</span>
                        <span class="font-semibold">${nft.price.toLocaleString()} LLT-AVATAR</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Collection</span>
                        <span>${nft.collection}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Rarity</span>
                        <span>${nft.rarity}</span>
                    </div>
                </div>

                <button
                    onclick="this.buyNFT(${nft.id})"
                    class="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 px-6 py-3 rounded-lg font-semibold transition-all">
                    Buy Now
                </button>
            </div>
        `;

        // Add buy NFT functionality
        modal.querySelector('button[onclick*="buyNFT"]').onclick = () => {
            MarketplaceManager.buyNFT(nft.id);
            modal.remove();
        };

        return modal;
    }
};

// Initialize marketplace when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize marketplace after a short delay to ensure other modules are ready
    setTimeout(() => {
        MarketplaceManager.initialize();
        NFTManager.initialize();
    }, 1000);
});

// Export for use in other modules
window.MarketplaceManager = MarketplaceManager;
window.NFTManager = NFTManager;