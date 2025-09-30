/**
 * LL TOKEN OFFLINE - Wallet Module
 * Quantum-safe wallet operations and transaction management
 */

// Wallet Management Functions
const WalletManager = {
    // Mock quantum wallet integration
    quantumWallet: null,

    // Initialize wallet connection
    async initializeWallet() {
        try {
            // Simulate quantum wallet initialization
            this.quantumWallet = {
                id: LLToken.walletId,
                publicKey: this.generateMockPublicKey(),
                encrypted: true,
                quantumSafe: true,
                offlineCapable: true
            };

            console.log('🔐 Quantum wallet initialized:', this.quantumWallet.id);
            return true;
        } catch (error) {
            console.error('❌ Wallet initialization failed:', error);
            return false;
        }
    },

    // Generate mock public key for demonstration
    generateMockPublicKey() {
        const chars = '0123456789ABCDEF';
        let publicKey = '';
        for (let i = 0; i < 64; i++) {
            publicKey += chars.charAt(Math.floor(Math.random() * chars.length));
        }
        return publicKey;
    },

    // Send tokens
    async sendTokens(toAddress, amount, tokenType) {
        try {
            LLTokenApp.showLoading('Creating quantum-safe transaction...');

            // Simulate quantum signature generation
            await this.simulateDelay(2000);

            const transaction = {
                id: `TX-${Date.now()}-${Math.random().toString(36).substr(2, 9).toUpperCase()}`,
                from: this.quantumWallet.id,
                to: toAddress,
                amount: parseFloat(amount),
                tokenType: tokenType,
                timestamp: new Date().toISOString(),
                signature: this.generateQuantumSignature(),
                status: 'pending',
                quantumSafe: true,
                offlineMode: true
            };

            // Add to pending transactions
            LLToken.transactions.unshift(transaction);

            // Update balance
            if (LLToken.tokenHoldings[tokenType]) {
                LLToken.tokenHoldings[tokenType].balance -= parseFloat(amount);
            }

            // Simulate confirmation after delay
            setTimeout(() => {
                transaction.status = 'confirmed';
                this.updateWalletUI();
                LLTokenApp.showToast(`Transaction confirmed: ${amount} ${tokenType} sent`, 'success');
            }, 5000);

            this.updateWalletUI();
            LLTokenApp.hideLoading();

            return transaction;
        } catch (error) {
            LLTokenApp.hideLoading();
            LLTokenApp.showToast('Transaction failed: ' + error.message, 'error');
            throw error;
        }
    },

    // Receive tokens (generate address)
    generateReceiveAddress(tokenType = 'LLT-AVATAR') {
        const address = {
            address: this.quantumWallet.id,
            publicKey: this.quantumWallet.publicKey,
            tokenType: tokenType,
            qrCode: this.generateQRCode(this.quantumWallet.id),
            quantumSafe: true
        };

        return address;
    },

    // Generate quantum-safe signature (mock)
    generateQuantumSignature() {
        const timestamp = Date.now().toString();
        const randomData = Math.random().toString(36);
        const mockSignature = btoa(timestamp + randomData).substr(0, 88); // Ed25519 signatures are 64 bytes = 88 base64 chars
        return mockSignature + '=='; // Pad to proper length
    },

    // Generate QR code data (mock)
    generateQRCode(data) {
        return `data:image/svg+xml;base64,${btoa(`
            <svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
                <rect width="200" height="200" fill="white"/>
                <text x="100" y="100" text-anchor="middle" fill="black" font-size="12">
                    QR Code: ${data.substr(0, 10)}...
                </text>
            </svg>
        `)}`;
    },

    // Swap tokens
    async swapTokens(fromToken, toToken, amount) {
        try {
            LLTokenApp.showLoading('Processing quantum-safe swap...');

            // Simulate swap calculation
            const swapRate = this.calculateSwapRate(fromToken, toToken);
            const receivedAmount = amount * swapRate;

            await this.simulateDelay(3000);

            // Create swap transaction
            const swapTransaction = {
                id: `SWAP-${Date.now()}`,
                type: 'swap',
                fromToken: fromToken,
                toToken: toToken,
                fromAmount: amount,
                toAmount: receivedAmount,
                rate: swapRate,
                timestamp: new Date().toISOString(),
                signature: this.generateQuantumSignature(),
                status: 'confirmed',
                quantumSafe: true
            };

            // Update balances
            if (LLToken.tokenHoldings[fromToken]) {
                LLToken.tokenHoldings[fromToken].balance -= amount;
            }
            if (LLToken.tokenHoldings[toToken]) {
                LLToken.tokenHoldings[toToken].balance += receivedAmount;
            }

            // Add to transactions
            LLToken.transactions.unshift(swapTransaction);

            this.updateWalletUI();
            LLTokenApp.hideLoading();
            LLTokenApp.showToast(`Swap completed: ${amount} ${fromToken} → ${receivedAmount.toFixed(2)} ${toToken}`, 'success');

            return swapTransaction;
        } catch (error) {
            LLTokenApp.hideLoading();
            LLTokenApp.showToast('Swap failed: ' + error.message, 'error');
            throw error;
        }
    },

    // Calculate swap rate (mock)
    calculateSwapRate(fromToken, toToken) {
        // Mock exchange rates based on token utility and scarcity
        const rates = {
            'LLT-GOV': 5.2,    // Governance tokens are valuable
            'LLT-LAND': 3.8,   // Land tokens are scarce
            'LLT-STAKE': 2.1,  // Staking tokens have utility
            'LLT-AVATAR': 1.0, // Base token
            'LLT-ASSET': 0.8,
            'LLT-COMPUTE': 1.2,
            'LLT-DATA': 0.9,
            'LLT-EXP': 0.3,    // Experience tokens are abundant
            'LLT-REP': 0.5,    // Reputation tokens are soul-bound
            'LLT-CREATE': 1.1
        };

        const fromRate = rates[fromToken] || 1.0;
        const toRate = rates[toToken] || 1.0;

        // Add some randomness for market dynamics
        const marketVariance = 0.9 + (Math.random() * 0.2); // ±10% variance

        return (fromRate / toRate) * marketVariance;
    },

    // Update wallet UI
    updateWalletUI() {
        // Recalculate total balance
        let totalBalance = 0;
        Object.values(LLToken.tokenHoldings).forEach(holding => {
            totalBalance += holding.balance;
        });
        LLToken.totalBalance = totalBalance;

        // Update total balance display
        const totalBalanceElement = document.getElementById('totalBalance');
        if (totalBalanceElement) {
            totalBalanceElement.textContent = totalBalance.toLocaleString();
        }

        // Update USD value
        const usdValueElement = document.getElementById('usdValue');
        if (usdValueElement) {
            const mockUsdValue = (totalBalance * 0.05).toFixed(2);
            usdValueElement.textContent = `(≈ $${mockUsdValue})`;
        }

        // Refresh token holdings
        if (typeof updateTokenHoldingsUI === 'function') {
            updateTokenHoldingsUI();
        }

        // Refresh transactions
        if (typeof updateTransactionsUI === 'function') {
            updateTransactionsUI();
        }
    },

    // Export wallet data
    exportWallet() {
        const walletData = {
            walletId: this.quantumWallet.id,
            publicKey: this.quantumWallet.publicKey,
            tokenHoldings: LLToken.tokenHoldings,
            transactions: LLToken.transactions,
            exportedAt: new Date().toISOString(),
            version: '1.0.0',
            quantumSafe: true
        };

        return walletData;
    },

    // Import wallet data
    importWallet(walletData) {
        try {
            // Validate wallet data
            if (!walletData.walletId || !walletData.quantumSafe) {
                throw new Error('Invalid wallet data or not quantum-safe');
            }

            // Restore wallet state
            LLToken.walletId = walletData.walletId;
            LLToken.tokenHoldings = walletData.tokenHoldings || {};
            LLToken.transactions = walletData.transactions || [];

            this.quantumWallet = {
                id: walletData.walletId,
                publicKey: walletData.publicKey,
                encrypted: true,
                quantumSafe: true,
                offlineCapable: true
            };

            this.updateWalletUI();
            LLTokenApp.showToast('Wallet imported successfully', 'success');

            return true;
        } catch (error) {
            LLTokenApp.showToast('Wallet import failed: ' + error.message, 'error');
            return false;
        }
    },

    // Backup wallet
    backupWallet() {
        const backup = this.exportWallet();
        const backupString = JSON.stringify(backup, null, 2);

        // Create download link
        const blob = new Blob([backupString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `lltoken-wallet-backup-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        URL.revokeObjectURL(url);

        LLTokenApp.showToast('Wallet backup downloaded', 'success');
    },

    // Simulate network delay
    simulateDelay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
};

// Transaction History Manager
const TransactionManager = {
    // Get transaction history with filters
    getTransactionHistory(filters = {}) {
        let transactions = [...LLToken.transactions];

        // Apply filters
        if (filters.tokenType) {
            transactions = transactions.filter(tx => tx.tokenType === filters.tokenType);
        }

        if (filters.type) {
            transactions = transactions.filter(tx => tx.type === filters.type);
        }

        if (filters.direction) {
            transactions = transactions.filter(tx => tx.direction === filters.direction);
        }

        if (filters.status) {
            transactions = transactions.filter(tx => tx.status === filters.status);
        }

        if (filters.dateFrom) {
            transactions = transactions.filter(tx => new Date(tx.timestamp) >= new Date(filters.dateFrom));
        }

        if (filters.dateTo) {
            transactions = transactions.filter(tx => new Date(tx.timestamp) <= new Date(filters.dateTo));
        }

        return transactions;
    },

    // Export transaction history
    exportTransactionHistory() {
        const history = {
            walletId: WalletManager.quantumWallet?.id,
            transactions: LLToken.transactions,
            exportedAt: new Date().toISOString(),
            totalTransactions: LLToken.transactions.length
        };

        const historyString = JSON.stringify(history, null, 2);
        const blob = new Blob([historyString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `lltoken-transaction-history-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        URL.revokeObjectURL(url);

        LLTokenApp.showToast('Transaction history exported', 'success');
    },

    // Get transaction statistics
    getTransactionStats() {
        const transactions = LLToken.transactions;

        const stats = {
            total: transactions.length,
            confirmed: transactions.filter(tx => tx.status === 'confirmed').length,
            pending: transactions.filter(tx => tx.status === 'pending').length,
            incoming: transactions.filter(tx => tx.direction === 'in').length,
            outgoing: transactions.filter(tx => tx.direction === 'out').length,
            byTokenType: {},
            totalVolume: 0
        };

        // Calculate volume and token type distribution
        transactions.forEach(tx => {
            if (tx.amount) {
                stats.totalVolume += tx.amount;
            }

            if (tx.tokenType) {
                stats.byTokenType[tx.tokenType] = (stats.byTokenType[tx.tokenType] || 0) + 1;
            }
        });

        return stats;
    }
};

// Event Listeners for Wallet Actions
document.addEventListener('DOMContentLoaded', function() {
    // Initialize wallet manager
    WalletManager.initializeWallet();

    // Send token modal
    setupSendTokenModal();

    // Receive token modal
    setupReceiveTokenModal();

    // Swap token modal
    setupSwapTokenModal();

    // Action button listeners
    setupActionButtons();
});

function setupSendTokenModal() {
    // Create and show send token modal when send button is clicked
    const sendButtons = document.querySelectorAll('.action-btn');
    if (sendButtons[0]) { // First button is send
        sendButtons[0].addEventListener('click', showSendTokenModal);
    }
}

function showSendTokenModal() {
    const modal = createModal('Send Tokens', `
        <form id="sendTokenForm" class="space-y-4">
            <div>
                <label class="block text-sm font-medium mb-2">Token Type</label>
                <select id="sendTokenType" class="w-full" required>
                    ${Object.keys(LLToken.tokenSpecs).map(token =>
                        `<option value="${token}">${LLToken.tokenSpecs[token].name}</option>`
                    ).join('')}
                </select>
            </div>
            <div>
                <label class="block text-sm font-medium mb-2">Recipient Address</label>
                <input type="text" id="sendToAddress" class="w-full" placeholder="WALLET_..." required>
            </div>
            <div>
                <label class="block text-sm font-medium mb-2">Amount</label>
                <input type="number" id="sendAmount" class="w-full" placeholder="0.00" min="0" step="0.000001" required>
            </div>
            <button type="submit" class="w-full bg-purple-500 hover:bg-purple-600 px-4 py-3 rounded-lg font-semibold transition-all">
                Send Tokens
            </button>
        </form>
    `);

    document.getElementById('sendTokenForm').addEventListener('submit', async (e) => {
        e.preventDefault();

        const tokenType = document.getElementById('sendTokenType').value;
        const toAddress = document.getElementById('sendToAddress').value;
        const amount = parseFloat(document.getElementById('sendAmount').value);

        try {
            await WalletManager.sendTokens(toAddress, amount, tokenType);
            closeModal();
        } catch (error) {
            console.error('Send failed:', error);
        }
    });
}

function setupReceiveTokenModal() {
    const sendButtons = document.querySelectorAll('.action-btn');
    if (sendButtons[1]) { // Second button is receive
        sendButtons[1].addEventListener('click', showReceiveTokenModal);
    }
}

function showReceiveTokenModal() {
    const address = WalletManager.generateReceiveAddress();

    const modal = createModal('Receive Tokens', `
        <div class="text-center space-y-4">
            <div class="bg-white p-4 rounded-lg">
                <img src="${address.qrCode}" alt="Wallet QR Code" class="mx-auto mb-4" width="150" height="150">
            </div>
            <div>
                <label class="block text-sm font-medium mb-2">Your Wallet Address</label>
                <div class="flex">
                    <input type="text" value="${address.address}" class="flex-1" readonly>
                    <button onclick="copyToClipboard('${address.address}')" class="ml-2 bg-blue-500 hover:bg-blue-600 px-4 py-2 rounded-lg">
                        Copy
                    </button>
                </div>
            </div>
            <p class="text-sm text-gray-400">
                Share this address to receive LL TOKEN payments
            </p>
        </div>
    `);
}

function setupSwapTokenModal() {
    const sendButtons = document.querySelectorAll('.action-btn');
    if (sendButtons[3]) { // Fourth button is swap
        sendButtons[3].addEventListener('click', showSwapTokenModal);
    }
}

function showSwapTokenModal() {
    const modal = createModal('Swap Tokens', `
        <form id="swapTokenForm" class="space-y-4">
            <div>
                <label class="block text-sm font-medium mb-2">From Token</label>
                <select id="swapFromToken" class="w-full" required>
                    ${Object.keys(LLToken.tokenSpecs).map(token =>
                        `<option value="${token}">${LLToken.tokenSpecs[token].name}</option>`
                    ).join('')}
                </select>
            </div>
            <div>
                <label class="block text-sm font-medium mb-2">Amount</label>
                <input type="number" id="swapAmount" class="w-full" placeholder="0.00" min="0" step="0.000001" required>
            </div>
            <div class="text-center">
                <button type="button" class="bg-gray-600 p-2 rounded-full">🔄</button>
            </div>
            <div>
                <label class="block text-sm font-medium mb-2">To Token</label>
                <select id="swapToToken" class="w-full" required>
                    ${Object.keys(LLToken.tokenSpecs).map(token =>
                        `<option value="${token}">${LLToken.tokenSpecs[token].name}</option>`
                    ).join('')}
                </select>
            </div>
            <div id="swapPreview" class="bg-white/5 p-4 rounded-lg hidden">
                <p class="text-sm text-gray-400 mb-1">You will receive approximately:</p>
                <p id="swapEstimate" class="text-lg font-semibold">0.00 LLT</p>
            </div>
            <button type="submit" class="w-full bg-green-500 hover:bg-green-600 px-4 py-3 rounded-lg font-semibold transition-all">
                Swap Tokens
            </button>
        </form>
    `);

    // Add swap calculation preview
    const amountInput = document.getElementById('swapAmount');
    const fromTokenSelect = document.getElementById('swapFromToken');
    const toTokenSelect = document.getElementById('swapToToken');

    function updateSwapPreview() {
        const amount = parseFloat(amountInput.value) || 0;
        const fromToken = fromTokenSelect.value;
        const toToken = toTokenSelect.value;

        if (amount > 0 && fromToken && toToken && fromToken !== toToken) {
            const rate = WalletManager.calculateSwapRate(fromToken, toToken);
            const estimate = amount * rate;

            document.getElementById('swapEstimate').textContent = `${estimate.toFixed(6)} ${toToken}`;
            document.getElementById('swapPreview').classList.remove('hidden');
        } else {
            document.getElementById('swapPreview').classList.add('hidden');
        }
    }

    amountInput.addEventListener('input', updateSwapPreview);
    fromTokenSelect.addEventListener('change', updateSwapPreview);
    toTokenSelect.addEventListener('change', updateSwapPreview);

    document.getElementById('swapTokenForm').addEventListener('submit', async (e) => {
        e.preventDefault();

        const fromToken = fromTokenSelect.value;
        const toToken = toTokenSelect.value;
        const amount = parseFloat(amountInput.value);

        if (fromToken === toToken) {
            LLTokenApp.showToast('Cannot swap the same token', 'error');
            return;
        }

        try {
            await WalletManager.swapTokens(fromToken, toToken, amount);
            closeModal();
        } catch (error) {
            console.error('Swap failed:', error);
        }
    });
}

function setupActionButtons() {
    const actionButtons = document.querySelectorAll('.action-btn');

    // Stake button (third button)
    if (actionButtons[2]) {
        actionButtons[2].addEventListener('click', () => {
            LLTokenApp.switchTab('staking', 'stakingSection');
        });
    }
}

// Utility Functions
function createModal(title, content) {
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50';
    modal.innerHTML = `
        <div class="bg-white/10 backdrop-blur-lg border border-purple-500/30 rounded-xl p-6 max-w-md w-full mx-4">
            <div class="flex justify-between items-center mb-6">
                <h3 class="text-xl font-semibold">${title}</h3>
                <button onclick="closeModal()" class="text-gray-400 hover:text-white text-2xl">&times;</button>
            </div>
            ${content}
        </div>
    `;

    document.body.appendChild(modal);
    return modal;
}

function closeModal() {
    const modal = document.querySelector('.fixed.inset-0.z-50');
    if (modal) {
        document.body.removeChild(modal);
    }
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        LLTokenApp.showToast('Address copied to clipboard', 'success');
    }).catch(() => {
        LLTokenApp.showToast('Failed to copy address', 'error');
    });
}

// Export wallet manager for use in other modules
window.WalletManager = WalletManager;
window.TransactionManager = TransactionManager;