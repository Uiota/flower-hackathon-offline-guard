/**
 * LL TOKEN OFFLINE - Metaverse Module
 * Avatar management, virtual worlds, and cross-world functionality
 */

const MetaverseManager = {
    // Current user data
    userData: {
        avatars: [],
        virtualAssets: [],
        worldConnections: {},
        reputation: 0.85,
        experience: 0,
        achievements: []
    },

    // Supported virtual worlds
    supportedWorlds: {
        'vrchat': {
            name: 'VRChat Universe',
            emoji: '🌐',
            status: 'connected',
            users: 1247,
            features: ['Avatar customization', 'Social interaction', 'World creation'],
            tokenUtilities: ['LLT-AVATAR', 'LLT-ASSET', 'LLT-REP', 'LLT-EXP']
        },
        'horizon': {
            name: 'Horizon Worlds',
            emoji: '🌅',
            status: 'available',
            users: 892,
            features: ['Meta integration', 'Enterprise spaces', 'VR meetings'],
            tokenUtilities: ['LLT-AVATAR', 'LLT-COLLAB', 'LLT-REP']
        },
        'decentraland': {
            name: 'Decentraland',
            emoji: '🏙️',
            status: 'connected',
            users: 634,
            features: ['Land ownership', 'NFT galleries', 'Virtual commerce'],
            tokenUtilities: ['LLT-LAND', 'LLT-ASSET', 'LLT-GOV']
        },
        'sandbox': {
            name: 'The Sandbox',
            emoji: '🏖️',
            status: 'available',
            users: 1103,
            features: ['Game creation', 'Asset marketplace', 'Community events'],
            tokenUtilities: ['LLT-CREATE', 'LLT-ASSET', 'LLT-LAND']
        },
        'somnium': {
            name: 'Somnium Space',
            emoji: '🚀',
            status: 'available',
            users: 423,
            features: ['VR experiences', 'Virtual real estate', 'Custom worlds'],
            tokenUtilities: ['LLT-LAND', 'LLT-AVATAR', 'LLT-ASSET']
        },
        'recroom': {
            name: 'Rec Room',
            emoji: '🎮',
            status: 'connected',
            users: 756,
            features: ['Social gaming', 'User-generated content', 'Cross-platform'],
            tokenUtilities: ['LLT-EXP', 'LLT-REP', 'LLT-COLLAB']
        }
    },

    // Initialize metaverse system
    initialize() {
        this.loadUserAvatar();
        this.loadVirtualAssets();
        this.loadWorldConnections();
        this.setupMetaverseInterface();
        this.startMetaverseUpdates();

        console.log('🌐 Metaverse system initialized');
    },

    // Load user avatar data
    loadUserAvatar() {
        // Generate or load existing avatar
        this.userData.avatars = [{
            id: 'AVT-QE001',
            name: 'Quantum Explorer',
            level: 15,
            attributes: {
                strength: 15,
                intelligence: 20,
                charisma: 12,
                quantum_affinity: 18
            },
            abilities: ['teleportation', 'quantum_sight', 'data_manipulation'],
            appearance: {
                skin_tone: 'digital_blue',
                hair_color: 'quantum_purple',
                eye_color: 'neon_cyan',
                style: 'cyberpunk'
            },
            equipment: {
                weapon: 'Quantum Blade',
                armor: 'Data Guardian Suit',
                accessory: 'Wisdom Crown'
            },
            experience: 24750,
            nextLevelAt: 25000,
            reputation: 0.87,
            createdDate: new Date('2024-01-15'),
            worldsVisited: ['vrchat', 'decentraland', 'horizon'],
            achievements: ['First Avatar', 'World Jumper', 'Quantum Master']
        }];

        this.updateAvatarUI();
    },

    // Update avatar UI
    updateAvatarUI() {
        const avatar = this.userData.avatars[0];
        if (!avatar) return;

        // Update avatar name and ID
        const avatarNameEl = document.getElementById('avatarName');
        const avatarIdEl = document.getElementById('avatarId');

        if (avatarNameEl) avatarNameEl.textContent = avatar.name;
        if (avatarIdEl) avatarIdEl.textContent = `Avatar ID: ${avatar.id}`;

        // Update attribute bars
        this.updateAttributeBar('strength', avatar.attributes.strength, 20);
        this.updateAttributeBar('intelligence', avatar.attributes.intelligence, 20);
        this.updateAttributeBar('charisma', avatar.attributes.charisma, 20);
    },

    // Update attribute progress bar
    updateAttributeBar(attribute, current, max) {
        const percentage = (current / max) * 100;

        // This would update the progress bars in the UI
        // For now, just log the values
        console.log(`${attribute}: ${current}/${max} (${percentage}%)`);
    },

    // Load virtual assets
    loadVirtualAssets() {
        this.userData.virtualAssets = [
            {
                id: 'ASSET-001',
                name: 'Quantum Blade',
                type: 'weapon',
                rarity: 'legendary',
                emoji: '⚔️',
                description: 'A blade that cuts through dimensions',
                attributes: {
                    damage: 95,
                    speed: 80,
                    quantum_power: 100
                },
                worldCompatibility: ['vrchat', 'decentraland', 'sandbox'],
                acquiredDate: new Date('2024-02-01'),
                tradeable: true,
                estimatedValue: 3200
            },
            {
                id: 'ASSET-002',
                name: 'Wisdom Crown',
                type: 'accessory',
                rarity: 'epic',
                emoji: '👑',
                description: 'Grants enhanced learning capabilities',
                attributes: {
                    intelligence_boost: 25,
                    experience_multiplier: 1.5,
                    reputation_bonus: 0.1
                },
                worldCompatibility: ['vrchat', 'horizon', 'recroom'],
                acquiredDate: new Date('2024-01-20'),
                tradeable: true,
                estimatedValue: 1800
            },
            {
                id: 'ASSET-003',
                name: 'Portal Stone',
                type: 'tool',
                rarity: 'rare',
                emoji: '🌀',
                description: 'Enables instant travel between worlds',
                attributes: {
                    teleport_range: 'unlimited',
                    cooldown: 60,
                    energy_cost: 'low'
                },
                worldCompatibility: ['all'],
                acquiredDate: new Date('2024-03-10'),
                tradeable: false, // Soul-bound
                estimatedValue: 2500
            },
            {
                id: 'ASSET-004',
                name: 'Energy Crystal',
                type: 'resource',
                rarity: 'common',
                emoji: '💎',
                description: 'Provides energy for various activities',
                attributes: {
                    energy_amount: 1000,
                    regeneration_rate: 10,
                    purity: 'high'
                },
                worldCompatibility: ['all'],
                acquiredDate: new Date('2024-03-15'),
                tradeable: true,
                estimatedValue: 150,
                quantity: 5
            }
        ];

        this.updateVirtualAssetsUI();
    },

    // Update virtual assets UI
    updateVirtualAssetsUI() {
        const assetsContainer = document.getElementById('virtualAssets');
        if (!assetsContainer) return;

        assetsContainer.innerHTML = '';

        this.userData.virtualAssets.forEach(asset => {
            const rarityColors = {
                'legendary': 'border-yellow-400/50 text-yellow-400',
                'epic': 'border-purple-400/50 text-purple-400',
                'rare': 'border-blue-400/50 text-blue-400',
                'common': 'border-gray-400/50 text-gray-400'
            };

            const assetCard = document.createElement('div');
            assetCard.className = `bg-white/5 border rounded-lg p-4 hover:bg-white/8 transition-all cursor-pointer ${rarityColors[asset.rarity]}`;
            assetCard.innerHTML = `
                <div class="text-3xl text-center mb-3">${asset.emoji}</div>
                <h4 class="font-medium text-center mb-1">${asset.name}</h4>
                <p class="text-xs text-gray-400 text-center mb-2">${asset.type}</p>
                <p class="text-xs text-center ${rarityColors[asset.rarity].split(' ')[1]}">${asset.rarity}</p>
                ${asset.quantity ? `<p class="text-xs text-center text-gray-300 mt-1">Qty: ${asset.quantity}</p>` : ''}
                <div class="mt-3 pt-3 border-t border-gray-600">
                    <p class="text-xs text-gray-400 mb-1">Value: ${asset.estimatedValue.toLocaleString()} LLT</p>
                    <p class="text-xs text-gray-300">${asset.worldCompatibility.length === 1 && asset.worldCompatibility[0] === 'all' ? 'Universal' : `${asset.worldCompatibility.length} worlds`}</p>
                </div>
            `;

            assetCard.addEventListener('click', () => this.showAssetDetails(asset.id));
            assetsContainer.appendChild(assetCard);
        });
    },

    // Load world connections
    loadWorldConnections() {
        // Initialize connection status for each world
        Object.keys(this.supportedWorlds).forEach(worldId => {
            const world = this.supportedWorlds[worldId];
            this.userData.worldConnections[worldId] = {
                connected: world.status === 'connected',
                lastVisit: world.status === 'connected' ? new Date(Date.now() - Math.random() * 86400000 * 7) : null,
                reputation: Math.random() * 0.5 + 0.5, // 0.5 to 1.0
                achievements: [],
                timeSpent: Math.floor(Math.random() * 100) + 10 // hours
            };
        });

        this.updateVirtualWorldsUI();
    },

    // Update virtual worlds UI
    updateVirtualWorldsUI() {
        const worldsContainer = document.getElementById('virtualWorlds');
        if (!worldsContainer) return;

        worldsContainer.innerHTML = '';

        Object.entries(this.supportedWorlds).forEach(([worldId, world]) => {
            const connection = this.userData.worldConnections[worldId];
            const statusColor = connection.connected ? 'text-green-400' : 'text-yellow-400';
            const statusDot = connection.connected ? 'bg-green-500' : 'bg-yellow-500';

            const worldCard = document.createElement('div');
            worldCard.className = 'flex items-center justify-between p-4 bg-white/5 border border-purple-500/20 rounded-lg hover:bg-white/8 transition-all cursor-pointer';
            worldCard.innerHTML = `
                <div class="flex items-center space-x-3">
                    <div class="text-2xl">${world.emoji}</div>
                    <div>
                        <p class="font-medium">${world.name}</p>
                        <p class="text-sm text-gray-400">${world.users} active users</p>
                        <p class="text-xs text-gray-500">${world.features.slice(0, 2).join(', ')}</p>
                    </div>
                </div>
                <div class="text-right">
                    <div class="flex items-center space-x-2 mb-1">
                        <div class="w-2 h-2 ${statusDot} rounded-full"></div>
                        <span class="text-sm ${statusColor}">${world.status}</span>
                    </div>
                    <p class="text-xs text-gray-400">Rep: ${(connection.reputation * 100).toFixed(0)}%</p>
                    <p class="text-xs text-gray-500">${connection.timeSpent}h played</p>
                </div>
            `;

            worldCard.addEventListener('click', () => this.showWorldDetails(worldId));
            worldsContainer.appendChild(worldCard);
        });
    },

    // Show world details modal
    showWorldDetails(worldId) {
        const world = this.supportedWorlds[worldId];
        const connection = this.userData.worldConnections[worldId];

        if (!world) return;

        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50';
        modal.innerHTML = `
            <div class="bg-white/10 backdrop-blur-lg border border-purple-500/30 rounded-xl p-6 max-w-2xl w-full mx-4">
                <div class="flex justify-between items-center mb-6">
                    <div class="flex items-center space-x-3">
                        <div class="text-3xl">${world.emoji}</div>
                        <h3 class="text-xl font-semibold">${world.name}</h3>
                    </div>
                    <button onclick="this.closest('.fixed').remove()" class="text-gray-400 hover:text-white text-2xl">&times;</button>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <h4 class="font-semibold mb-3">World Features</h4>
                        <ul class="space-y-2">
                            ${world.features.map(feature => `
                                <li class="flex items-center space-x-2">
                                    <div class="w-2 h-2 bg-green-500 rounded-full"></div>
                                    <span class="text-sm">${feature}</span>
                                </li>
                            `).join('')}
                        </ul>
                    </div>

                    <div>
                        <h4 class="font-semibold mb-3">Your Stats</h4>
                        <div class="space-y-2">
                            <div class="flex justify-between">
                                <span class="text-sm text-gray-400">Status</span>
                                <span class="text-sm ${connection.connected ? 'text-green-400' : 'text-yellow-400'}">
                                    ${world.status}
                                </span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-sm text-gray-400">Reputation</span>
                                <span class="text-sm">${(connection.reputation * 100).toFixed(0)}%</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-sm text-gray-400">Time Spent</span>
                                <span class="text-sm">${connection.timeSpent} hours</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-sm text-gray-400">Last Visit</span>
                                <span class="text-sm">${connection.lastVisit ? connection.lastVisit.toLocaleDateString() : 'Never'}</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="mt-6">
                    <h4 class="font-semibold mb-3">Supported Token Utilities</h4>
                    <div class="flex flex-wrap gap-2">
                        ${world.tokenUtilities.map(token => `
                            <span class="px-3 py-1 bg-purple-500/20 text-purple-300 rounded-full text-sm">
                                ${token}
                            </span>
                        `).join('')}
                    </div>
                </div>

                <div class="mt-6 flex space-x-4">
                    ${connection.connected ?
                        `<button onclick="this.visitWorld('${worldId}')" class="flex-1 bg-green-500/20 border border-green-500/30 py-3 px-6 rounded-lg hover:bg-green-500/30 transition-all font-semibold">
                            Enter World
                        </button>` :
                        `<button onclick="this.connectToWorld('${worldId}')" class="flex-1 bg-blue-500/20 border border-blue-500/30 py-3 px-6 rounded-lg hover:bg-blue-500/30 transition-all font-semibold">
                            Connect
                        </button>`
                    }
                    <button onclick="this.manageAssets('${worldId}')" class="flex-1 bg-purple-500/20 border border-purple-500/30 py-3 px-6 rounded-lg hover:bg-purple-500/30 transition-all font-semibold">
                        Manage Assets
                    </button>
                </div>
            </div>
        `;

        // Add functionality to buttons
        modal.querySelector('button[onclick*="visitWorld"]')?.addEventListener('click', () => {
            this.visitWorld(worldId);
            modal.remove();
        });

        modal.querySelector('button[onclick*="connectToWorld"]')?.addEventListener('click', () => {
            this.connectToWorld(worldId);
            modal.remove();
        });

        modal.querySelector('button[onclick*="manageAssets"]')?.addEventListener('click', () => {
            this.manageWorldAssets(worldId);
            modal.remove();
        });

        document.body.appendChild(modal);
    },

    // Visit a world
    async visitWorld(worldId) {
        try {
            LLTokenApp.showLoading('Entering virtual world...');

            const world = this.supportedWorlds[worldId];
            const connection = this.userData.worldConnections[worldId];

            if (!connection.connected) {
                throw new Error('Not connected to this world. Please connect first.');
            }

            // Simulate world entry process
            await new Promise(resolve => setTimeout(resolve, 3000));

            // Update connection data
            connection.lastVisit = new Date();
            connection.timeSpent += Math.floor(Math.random() * 2) + 1; // 1-2 hours

            // Add experience
            const expGained = Math.floor(Math.random() * 100) + 50;
            this.userData.experience += expGained;

            // Update avatar in current world
            const avatar = this.userData.avatars[0];
            if (avatar && !avatar.worldsVisited.includes(worldId)) {
                avatar.worldsVisited.push(worldId);
            }

            LLTokenApp.hideLoading();
            LLTokenApp.showToast(`Successfully entered ${world.name}! Gained ${expGained} EXP`, 'success');

            // Update UI
            this.updateVirtualWorldsUI();

        } catch (error) {
            LLTokenApp.hideLoading();
            LLTokenApp.showToast(`Failed to enter world: ${error.message}`, 'error');
        }
    },

    // Connect to a world
    async connectToWorld(worldId) {
        try {
            LLTokenApp.showLoading('Connecting to virtual world...');

            const world = this.supportedWorlds[worldId];

            // Simulate connection process
            await new Promise(resolve => setTimeout(resolve, 2000));

            // Update connection status
            this.userData.worldConnections[worldId].connected = true;
            this.userData.worldConnections[worldId].lastVisit = new Date();
            this.supportedWorlds[worldId].status = 'connected';

            LLTokenApp.hideLoading();
            LLTokenApp.showToast(`Successfully connected to ${world.name}!`, 'success');

            // Update UI
            this.updateVirtualWorldsUI();

        } catch (error) {
            LLTokenApp.hideLoading();
            LLTokenApp.showToast(`Connection failed: ${error.message}`, 'error');
        }
    },

    // Manage assets for a specific world
    manageWorldAssets(worldId) {
        const world = this.supportedWorlds[worldId];
        const compatibleAssets = this.userData.virtualAssets.filter(asset =>
            asset.worldCompatibility.includes('all') || asset.worldCompatibility.includes(worldId)
        );

        LLTokenApp.showToast(`Managing assets for ${world.name}: ${compatibleAssets.length} compatible assets`, 'info');
    },

    // Show asset details modal
    showAssetDetails(assetId) {
        const asset = this.userData.virtualAssets.find(a => a.id === assetId);
        if (!asset) return;

        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50';
        modal.innerHTML = `
            <div class="bg-white/10 backdrop-blur-lg border border-purple-500/30 rounded-xl p-6 max-w-lg w-full mx-4">
                <div class="flex justify-between items-center mb-6">
                    <h3 class="text-xl font-semibold">${asset.name}</h3>
                    <button onclick="this.closest('.fixed').remove()" class="text-gray-400 hover:text-white text-2xl">&times;</button>
                </div>

                <div class="text-center mb-6">
                    <div class="text-6xl mb-4">${asset.emoji}</div>
                    <h4 class="text-lg font-medium mb-2">${asset.name}</h4>
                    <p class="text-sm text-gray-400 mb-2">${asset.type} • ${asset.rarity}</p>
                    <p class="text-sm text-gray-300">${asset.description}</p>
                </div>

                <div class="space-y-4 mb-6">
                    <div>
                        <h5 class="font-medium mb-2">Attributes</h5>
                        <div class="grid grid-cols-2 gap-2 text-sm">
                            ${Object.entries(asset.attributes).map(([key, value]) => `
                                <div class="flex justify-between">
                                    <span class="text-gray-400">${key.replace('_', ' ')}</span>
                                    <span>${value}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>

                    <div>
                        <h5 class="font-medium mb-2">Compatibility</h5>
                        <div class="flex flex-wrap gap-2">
                            ${asset.worldCompatibility.map(world => `
                                <span class="px-2 py-1 bg-blue-500/20 text-blue-300 rounded text-xs">
                                    ${world === 'all' ? 'Universal' : world}
                                </span>
                            `).join('')}
                        </div>
                    </div>

                    <div class="flex justify-between">
                        <span class="text-gray-400">Estimated Value</span>
                        <span class="font-semibold">${asset.estimatedValue.toLocaleString()} LLT</span>
                    </div>

                    <div class="flex justify-between">
                        <span class="text-gray-400">Acquired</span>
                        <span>${asset.acquiredDate.toLocaleDateString()}</span>
                    </div>

                    <div class="flex justify-between">
                        <span class="text-gray-400">Tradeable</span>
                        <span class="${asset.tradeable ? 'text-green-400' : 'text-red-400'}">
                            ${asset.tradeable ? 'Yes' : 'No (Soul-bound)'}
                        </span>
                    </div>
                </div>

                <div class="flex space-x-4">
                    ${asset.tradeable ?
                        `<button onclick="this.tradeAsset('${asset.id}')" class="flex-1 bg-purple-500/20 border border-purple-500/30 py-3 px-6 rounded-lg hover:bg-purple-500/30 transition-all font-semibold">
                            Trade Asset
                        </button>` :
                        `<button disabled class="flex-1 bg-gray-500/20 border border-gray-500/30 py-3 px-6 rounded-lg font-semibold opacity-50 cursor-not-allowed">
                            Soul-bound
                        </button>`
                    }
                    <button onclick="this.useAsset('${asset.id}')" class="flex-1 bg-green-500/20 border border-green-500/30 py-3 px-6 rounded-lg hover:bg-green-500/30 transition-all font-semibold">
                        Use Asset
                    </button>
                </div>
            </div>
        `;

        // Add functionality
        if (asset.tradeable) {
            modal.querySelector('button[onclick*="tradeAsset"]')?.addEventListener('click', () => {
                this.tradeAsset(asset.id);
                modal.remove();
            });
        }

        modal.querySelector('button[onclick*="useAsset"]')?.addEventListener('click', () => {
            this.useAsset(asset.id);
            modal.remove();
        });

        document.body.appendChild(modal);
    },

    // Trade an asset
    tradeAsset(assetId) {
        const asset = this.userData.virtualAssets.find(a => a.id === assetId);
        if (!asset) return;

        LLTokenApp.showToast(`Listed ${asset.name} for trade at ${asset.estimatedValue.toLocaleString()} LLT`, 'info');
    },

    // Use an asset
    useAsset(assetId) {
        const asset = this.userData.virtualAssets.find(a => a.id === assetId);
        if (!asset) return;

        LLTokenApp.showToast(`Using ${asset.name}... Effects applied!`, 'success');
    },

    // Setup metaverse interface
    setupMetaverseInterface() {
        // Avatar customization button
        const customizeBtn = document.querySelector('button:contains("Customize Avatar")');
        if (customizeBtn) {
            customizeBtn.addEventListener('click', this.showAvatarCustomization.bind(this));
        }

        // Skill upgrade button
        const upgradeBtn = document.querySelector('button:contains("Upgrade Skills")');
        if (upgradeBtn) {
            upgradeBtn.addEventListener('click', this.showSkillUpgrade.bind(this));
        }
    },

    // Show avatar customization interface
    showAvatarCustomization() {
        LLTokenApp.showToast('Avatar customization interface would open here', 'info');
        // This would open a detailed avatar editor
    },

    // Show skill upgrade interface
    showSkillUpgrade() {
        LLTokenApp.showToast('Skill upgrade interface would open here', 'info');
        // This would show available skill upgrades using LLT-EXP tokens
    },

    // Start periodic metaverse updates
    startMetaverseUpdates() {
        setInterval(() => {
            this.updateWorldStatuses();
            this.checkForAchievements();
        }, 30000); // Update every 30 seconds
    },

    // Update world statuses
    updateWorldStatuses() {
        Object.values(this.supportedWorlds).forEach(world => {
            // Simulate user count changes
            const change = Math.floor(Math.random() * 200) - 100; // ±100 users
            world.users = Math.max(0, world.users + change);
        });

        // Occasionally update connection statuses
        if (Math.random() < 0.1) { // 10% chance
            this.updateVirtualWorldsUI();
        }
    },

    // Check for achievements
    checkForAchievements() {
        const avatar = this.userData.avatars[0];
        if (!avatar) return;

        // Check for new achievements
        const achievements = [
            {
                id: 'world_hopper',
                name: 'World Hopper',
                condition: () => avatar.worldsVisited.length >= 3,
                reward: { type: 'LLT-EXP', amount: 500 }
            },
            {
                id: 'experienced_explorer',
                name: 'Experienced Explorer',
                condition: () => this.userData.experience >= 50000,
                reward: { type: 'LLT-REP', amount: 1000 }
            }
        ];

        achievements.forEach(achievement => {
            if (!avatar.achievements.includes(achievement.id) && achievement.condition()) {
                avatar.achievements.push(achievement.id);
                this.userData.achievements.push({
                    ...achievement,
                    earnedDate: new Date()
                });

                LLTokenApp.showToast(`Achievement unlocked: ${achievement.name}!`, 'success');
            }
        });
    }
};

// Cross-world asset bridge
const CrossWorldBridge = {
    // Initialize bridge system
    initialize() {
        console.log('🌉 Cross-world bridge initialized');
    },

    // Transfer asset between worlds
    async transferAsset(assetId, fromWorld, toWorld) {
        try {
            LLTokenApp.showLoading('Bridging asset between worlds...');

            const asset = MetaverseManager.userData.virtualAssets.find(a => a.id === assetId);
            if (!asset) {
                throw new Error('Asset not found');
            }

            // Check compatibility
            if (!asset.worldCompatibility.includes('all') &&
                !asset.worldCompatibility.includes(toWorld)) {
                throw new Error(`Asset not compatible with ${toWorld}`);
            }

            // Simulate bridge process
            await new Promise(resolve => setTimeout(resolve, 5000));

            // Calculate bridge fee (1% of asset value)
            const bridgeFee = Math.floor(asset.estimatedValue * 0.01);

            // Check if user has enough tokens for fee
            if (LLToken.tokenHoldings['LLT-ASSET']?.balance < bridgeFee) {
                throw new Error(`Insufficient LLT-ASSET tokens for bridge fee (${bridgeFee} required)`);
            }

            // Deduct bridge fee
            LLToken.tokenHoldings['LLT-ASSET'].balance -= bridgeFee;

            // Create bridge transaction
            const bridgeTransaction = {
                id: `BRIDGE-${Date.now()}`,
                type: 'Asset Bridge',
                assetId: asset.id,
                assetName: asset.name,
                fromWorld: fromWorld,
                toWorld: toWorld,
                fee: bridgeFee,
                timestamp: new Date(),
                status: 'completed'
            };

            // Add to transaction history
            LLToken.transactions.unshift({
                id: bridgeTransaction.id,
                type: bridgeTransaction.type,
                tokenType: 'LLT-ASSET',
                amount: -bridgeFee,
                direction: 'out',
                timestamp: bridgeTransaction.timestamp,
                status: 'confirmed',
                metadata: {
                    assetName: asset.name,
                    fromWorld: fromWorld,
                    toWorld: toWorld
                }
            });

            LLTokenApp.hideLoading();
            LLTokenApp.showToast(`Successfully bridged ${asset.name} from ${fromWorld} to ${toWorld}`, 'success');

            return bridgeTransaction;

        } catch (error) {
            LLTokenApp.hideLoading();
            LLTokenApp.showToast(`Bridge failed: ${error.message}`, 'error');
            throw error;
        }
    }
};

// Initialize metaverse when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        MetaverseManager.initialize();
        CrossWorldBridge.initialize();
    }, 1500);
});

// Export for use in other modules
window.MetaverseManager = MetaverseManager;
window.CrossWorldBridge = CrossWorldBridge;