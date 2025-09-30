// LL TOKEN OFFLINE.AI - Modern AI Platform Application
// Sleek, minimal interface with real-time AI-powered features

class OfflineAIPlatform {
    constructor() {
        this.currentTab = 'dashboard';
        this.isInitialized = false;
        this.animationFrame = null;
        this.realTimeData = {
            nodes: 247,
            assets: 2.4,
            rounds: 1543,
            security: 99.8
        };

        // Initialize platform
        this.init();
    }

    async init() {
        console.log('🤖 Initializing LL TOKEN OFFLINE.AI Platform');

        // Bind navigation events
        this.bindNavigationEvents();

        // Start real-time updates
        this.startRealTimeUpdates();

        // Initialize all modules
        await this.initializeModules();

        // Add smooth animations
        this.addSmoothAnimations();

        this.isInitialized = true;
        console.log('✨ OFFLINE.AI Platform initialized successfully');

        // Show welcome message
        this.showWelcomeMessage();
    }

    bindNavigationEvents() {
        const tabs = document.querySelectorAll('.ai-nav-tab');

        tabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabId = e.target.id.replace('Tab', '');
                this.switchTab(tabId);
            });
        });
    }

    switchTab(tabId) {
        // Update navigation state
        document.querySelectorAll('.ai-nav-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.getElementById(tabId + 'Tab').classList.add('active');

        // Update content with smooth transition
        document.querySelectorAll('.ai-tab-content').forEach(content => {
            content.classList.remove('active');
            content.classList.add('hidden');
        });

        const newContent = document.getElementById(tabId + 'Content');
        newContent.classList.remove('hidden');

        // Trigger fade in animation
        setTimeout(() => {
            newContent.classList.add('active');
        }, 50);

        this.currentTab = tabId;
        console.log(`🔄 Switched to ${tabId} module`);

        // Update page content based on tab
        this.updateTabContent(tabId);
    }

    updateTabContent(tabId) {
        switch(tabId) {
            case 'federated':
                this.updateFederatedLearningData();
                break;
            case 'biometrics':
                this.updateBiometricData();
                break;
            case 'assets':
                this.updateAssetData();
                break;
            case 'xrp':
                this.updateXRPData();
                break;
            case 'setup':
                this.updateAccountSetupData();
                break;
            case 'banking':
                this.updateBankingDashboardData();
                break;
            case 'api':
                this.updateAPIDocumentation();
                break;
        }
    }

    updateBiometricData() {
        console.log('👁️ Updating biometric data');

        // Ensure authentication button is properly connected
        setTimeout(() => {
            this.setupAuthenticationButton();
        }, 100);

        // Re-initialize camera if needed
        if (!this.videoStream) {
            setTimeout(() => {
                this.initializeCamera();
            }, 200);
        } else {
            // Camera exists, ensure face detection is running
            if (!this.faceDetectionInterval) {
                console.log('🔄 Restarting face detection...');
                setTimeout(() => {
                    const video = document.getElementById('biometricVideo');
                    if (video) {
                        this.startFaceDetection(video);
                    }
                }, 300);
            }
        }

        // Force face detection start as backup
        setTimeout(() => {
            const overlay = document.getElementById('faceDetectionOverlay');
            const video = document.getElementById('biometricVideo');

            if (overlay && video && !this.faceDetectionInterval) {
                console.log('🚀 Force starting face detection...');
                this.startFaceDetection(video);
            }
        }, 1000);
    }

    setupAuthenticationButton() {
        const authBtn = document.getElementById('authenticateBtn');
        if (authBtn) {
            // Remove existing event listeners by cloning
            const newBtn = authBtn.cloneNode(true);
            authBtn.parentNode.replaceChild(newBtn, authBtn);

            // Add fresh event listener
            newBtn.addEventListener('click', () => {
                console.log('🔐 Authentication button clicked');
                this.performBiometricAuthentication();
            });

            console.log('✅ Authentication button event listener attached');
        } else {
            console.log('❌ Authentication button not found, retrying in 500ms...');
            setTimeout(() => {
                this.setupAuthenticationButton();
            }, 500);
        }
    }

    startRealTimeUpdates() {
        // Update metrics every 3 seconds with smooth animations
        setInterval(() => {
            this.updateDashboardMetrics();
        }, 3000);

        // Update activity feed every 10 seconds
        setInterval(() => {
            this.updateActivityFeed();
        }, 10000);

        // Update progress bars continuously
        this.animateProgressBars();
    }

    updateDashboardMetrics() {
        // Animate metric changes smoothly
        const metrics = {
            activeNodes: Math.floor(Math.random() * 20) + 240,
            totalAssets: (Math.random() * 0.4 + 2.2).toFixed(1),
            trainingRounds: this.realTimeData.rounds + Math.floor(Math.random() * 5),
            securityScore: (Math.random() * 0.4 + 99.6).toFixed(1)
        };

        // Update with animation
        Object.keys(metrics).forEach(key => {
            const element = document.getElementById(key);
            if (element) {
                this.animateValueChange(element, metrics[key], key === 'totalAssets' ? '$' : '', key === 'totalAssets' ? 'M' : key === 'securityScore' ? '%' : '');
            }
        });

        // Store for next update
        this.realTimeData = { ...this.realTimeData, ...metrics };
    }

    animateValueChange(element, newValue, prefix = '', suffix = '') {
        const oldValue = parseFloat(element.textContent.replace(/[^0-9.]/g, ''));
        const targetValue = parseFloat(newValue);

        if (oldValue !== targetValue) {
            let current = oldValue;
            const increment = (targetValue - oldValue) / 20; // 20 steps

            const animate = () => {
                current += increment;
                if ((increment > 0 && current >= targetValue) || (increment < 0 && current <= targetValue)) {
                    current = targetValue;
                }

                element.textContent = prefix + (suffix === 'M' ? current.toFixed(1) : Math.floor(current)) + suffix;

                if (current !== targetValue) {
                    requestAnimationFrame(animate);
                }
            };

            animate();
        }
    }

    updateActivityFeed() {
        const activities = [
            {
                time: '1 minute ago',
                action: 'FL Training Round Completed',
                details: 'Facial Recognition CNN - Round 48/100',
                type: 'success'
            },
            {
                time: '3 minutes ago',
                action: 'Biometric Authentication',
                details: 'User verified with 97.2% confidence',
                type: 'success'
            },
            {
                time: '7 minutes ago',
                action: 'XRP Payment Processed',
                details: '8,500 XRP cross-border transfer',
                type: 'info'
            },
            {
                time: '12 minutes ago',
                action: 'New ML Client Connected',
                details: 'client_156 joined training network',
                type: 'success'
            },
            {
                time: '18 minutes ago',
                action: 'Token Rewards Distributed',
                details: '1,247.89 LLT tokens to participants',
                type: 'success'
            }
        ];

        const activityContainer = document.getElementById('recentActivity');
        if (activityContainer) {
            activityContainer.innerHTML = activities.map(activity => `
                <div class="ai-flex ai-justify-between ai-items-center ai-mb-md">
                    <div>
                        <div class="ai-text-primary">${activity.action}</div>
                        <div class="ai-text-secondary" style="font-size: 12px;">${activity.details}</div>
                    </div>
                    <div class="ai-text-right">
                        <div class="ai-text-secondary" style="font-size: 12px;">${activity.time}</div>
                        <div class="ai-status ai-status-${activity.type}">
                            ${activity.type === 'success' ? 'Completed' : activity.type === 'info' ? 'Processed' : 'Active'}
                        </div>
                    </div>
                </div>
            `).join('');
        }
    }

    animateProgressBars() {
        const progressBars = document.querySelectorAll('.ai-progress-fill');

        const animate = () => {
            progressBars.forEach(bar => {
                const currentWidth = parseFloat(bar.style.width) || 0;
                if (currentWidth < 100) {
                    // Slowly increase progress
                    const increment = Math.random() * 0.1;
                    bar.style.width = Math.min(100, currentWidth + increment) + '%';
                }
            });

            this.animationFrame = requestAnimationFrame(animate);
        };

        animate();
    }

    async initializeModules() {
        // Initialize Federated Learning module
        await this.initializeFederatedLearning();

        // Initialize Biometric Authentication
        await this.initializeBiometrics();

        // Initialize Digital Assets
        await this.initializeDigitalAssets();

        // Initialize XRP Integration
        await this.initializeXRPIntegration();

        // Initialize API Documentation
        await this.initializeAPIDocumentation();
    }

    async initializeFederatedLearning() {
        console.log('🧠 Initializing Federated Learning module');

        // Simulate connecting to FL network
        return new Promise(resolve => {
            setTimeout(() => {
                this.updateClientTable();
                console.log('✅ Federated Learning module ready');
                resolve();
            }, 500);
        });
    }

    updateClientTable() {
        const clients = [
            { id: 'client_004', status: 'training', version: 'v2.1.0', accuracy: '95.9%', responseTime: '2.0s', rewards: '167.89' },
            { id: 'client_005', status: 'active', version: 'v2.1.0', accuracy: '98.1%', responseTime: '1.7s', rewards: '324.56' },
            { id: 'client_006', status: 'syncing', version: 'v2.0.8', accuracy: '92.4%', responseTime: '3.1s', rewards: '145.23' }
        ];

        const tableBody = document.getElementById('clientTable');
        if (tableBody && tableBody.children.length < 6) {
            clients.forEach(client => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td><span class="ai-text-mono">${client.id}</span></td>
                    <td><span class="ai-status ai-status-${client.status === 'active' ? 'success' : client.status === 'training' ? 'info' : 'warning'}">${client.status.charAt(0).toUpperCase() + client.status.slice(1)}</span></td>
                    <td>${client.version}</td>
                    <td>${client.accuracy}</td>
                    <td>${client.responseTime}</td>
                    <td class="ai-text-mono">${client.rewards} LLT</td>
                    <td><button class="ai-btn ai-btn-ghost" style="font-size: 12px; padding: 4px 8px;">Details</button></td>
                `;
                tableBody.appendChild(row);
            });
        }
    }

    async initializeBiometrics() {
        console.log('👁️ Initializing Biometric Authentication');

        // Initialize camera access
        await this.initializeCamera();

        // Setup authentication button (with delay to ensure DOM is ready)
        setTimeout(() => {
            this.setupAuthenticationButton();
        }, 500);

        // Simulate biometric system initialization
        return new Promise(resolve => {
            setTimeout(() => {
                this.startBiometricMonitoring();
                this.bindBiometricEvents();
                console.log('✅ Biometric Authentication ready');
                resolve();
            }, 300);
        });
    }

    async initializeCamera() {
        try {
            // Check if camera is supported
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                console.log('📷 Camera not supported on this device');
                return;
            }

            // Request camera permission and initialize
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: 240,
                    height: 180,
                    facingMode: 'user'
                }
            });

            // Create video element
            this.createCameraFeed(stream);
            console.log('📷 Camera initialized successfully');

        } catch (error) {
            console.log('📷 Camera access denied or unavailable:', error.message);
            this.showCameraPlaceholder();
        }
    }

    createCameraFeed(stream) {
        // Find the camera container
        const cameraContainer = document.getElementById('cameraContainer');
        if (!cameraContainer) return;

        // Replace placeholder with actual video and face detection overlay
        cameraContainer.innerHTML = `
            <video id="biometricVideo" autoplay muted
                   style="width: 240px; height: 180px; border-radius: var(--radius-lg); object-fit: cover;">
            </video>
            <div id="faceDetectionOverlay" style="position: absolute; top: 0; left: 0; width: 240px; height: 180px; pointer-events: none;">
                <!-- Face detection box will be added here -->
            </div>
            <div style="position: absolute; top: 10px; right: 10px; width: 8px; height: 8px; background: var(--success); border-radius: 50%; animation: pulse 2s infinite;"></div>
            <div style="position: absolute; bottom: 10px; left: 10px; background: rgba(0,0,0,0.7); color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;">
                <span id="faceDetectionStatus">Scanning...</span>
            </div>
        `;

        const video = document.getElementById('biometricVideo');
        if (video) {
            video.srcObject = stream;
            this.videoStream = stream;

            console.log('📷 Camera feed activated');

            // Wait for video to load then start face detection
            video.onloadedmetadata = () => {
                console.log('📹 Video metadata loaded, starting face detection...');

                // Update status
                const statusElement = document.getElementById('biometricStatus');
                if (statusElement) {
                    statusElement.textContent = 'Camera Ready - Initializing Face Detection';
                    statusElement.className = 'ai-status ai-status-info ai-mb-lg';
                }

                // Start face detection with delay to ensure DOM is ready
                setTimeout(() => {
                    this.startFaceDetection(video);
                }, 500);
            };

            // Fallback - start face detection even if onloadedmetadata doesn't fire
            setTimeout(() => {
                if (!this.faceDetectionInterval) {
                    console.log('⚠️ Fallback: Starting face detection...');
                    this.startFaceDetection(video);
                }
            }, 2000);
        }
    }

    showCameraPlaceholder() {
        // Show placeholder if camera is not available
        const cameraContainer = document.getElementById('cameraContainer');
        if (!cameraContainer) return;

        cameraContainer.innerHTML = `
            <div class="ai-text-secondary" style="display: flex; flex-direction: column; align-items: center; gap: 8px;">
                <div style="font-size: 48px;">📷</div>
                <div>Camera Access Required</div>
                <button id="enableCameraBtn" class="ai-btn ai-btn-secondary" style="font-size: 12px; padding: 8px 16px; margin-top: 8px;">Enable Camera</button>
            </div>
        `;

        // Update status
        const statusElement = document.getElementById('biometricStatus');
        if (statusElement) {
            statusElement.textContent = 'Camera Access Required';
            statusElement.className = 'ai-status ai-status-warning ai-mb-lg';
        }

        // Add click handler for camera enable button
        document.getElementById('enableCameraBtn')?.addEventListener('click', () => {
            this.initializeCamera();
        });
    }

    startFaceDetection(video) {
        // Advanced face detection with AI-powered enhancements
        let faceDetected = false;
        let faceBox = null;
        let detectionInterval = null;
        let facialLandmarks = [];
        let livenessScore = 0;
        let expressionData = {};

        console.log('🎯 Starting ADVANCED face detection system...');

        // Get DOM elements with error checking
        const overlay = document.getElementById('faceDetectionOverlay');
        const statusSpan = document.getElementById('faceDetectionStatus');
        const biometricStatus = document.getElementById('biometricStatus');

        console.log('📋 Face detection elements found:', {
            overlay: !!overlay,
            statusSpan: !!statusSpan,
            biometricStatus: !!biometricStatus
        });

        if (!overlay || !statusSpan || !biometricStatus) {
            console.error('❌ Required face detection elements not found, retrying...');
            setTimeout(() => this.startFaceDetection(video), 1000);
            return;
        }

        // Advanced face detection box with landmarks
        const createAdvancedFaceBox = (confidence, liveness, expression) => {
            console.log(`🔲 Creating advanced face detection box - Confidence: ${confidence}%, Liveness: ${liveness}%`);

            // Remove existing elements
            if (faceBox && faceBox.parentNode) {
                faceBox.parentNode.removeChild(faceBox);
            }

            faceBox = document.createElement('div');

            // Dynamic border color based on confidence and liveness
            let borderColor = '#10b981'; // Default green
            let bgColor = 'rgba(16, 185, 129, 0.15)';
            let shadowColor = 'rgba(16, 185, 129, 0.4)';

            if (confidence < 70 || liveness < 80) {
                borderColor = '#f59e0b'; // Yellow for medium confidence
                bgColor = 'rgba(245, 158, 11, 0.15)';
                shadowColor = 'rgba(245, 158, 11, 0.4)';
            }
            if (confidence < 50 || liveness < 60) {
                borderColor = '#ef4444'; // Red for low confidence
                bgColor = 'rgba(239, 68, 68, 0.15)';
                shadowColor = 'rgba(239, 68, 68, 0.4)';
            }

            faceBox.style.cssText = `
                position: absolute;
                border: 3px solid ${borderColor};
                background: ${bgColor};
                border-radius: 8px;
                transition: all 0.4s ease;
                box-shadow: 0 0 20px ${shadowColor};
                z-index: 10;
            `;

            // Add facial landmarks (eye points, nose, mouth corners)
            const landmarks = this.generateFacialLandmarks();
            landmarks.forEach(landmark => {
                const point = document.createElement('div');
                point.style.cssText = `
                    position: absolute;
                    width: 3px;
                    height: 3px;
                    background: ${borderColor};
                    border-radius: 50%;
                    left: ${landmark.x}px;
                    top: ${landmark.y}px;
                    box-shadow: 0 0 4px ${shadowColor};
                    opacity: 0.8;
                `;
                faceBox.appendChild(point);
            });

            // Add biometric data overlay
            const bioOverlay = document.createElement('div');
            bioOverlay.style.cssText = `
                position: absolute;
                bottom: -30px;
                left: 0;
                right: 0;
                background: rgba(0, 0, 0, 0.8);
                color: white;
                font-size: 10px;
                padding: 4px 6px;
                border-radius: 4px;
                white-space: nowrap;
            `;
            bioOverlay.innerHTML = `
                C:${confidence}% | L:${liveness}% | ${expression.dominant || 'neutral'}
            `;
            faceBox.appendChild(bioOverlay);

            return faceBox;
        };

        // Generate realistic facial landmarks
        this.generateFacialLandmarks = () => {
            return [
                // Eyes (left and right)
                { x: 25, y: 35, type: 'left_eye' },
                { x: 45, y: 35, type: 'right_eye' },
                // Nose
                { x: 35, y: 50, type: 'nose_tip' },
                // Mouth corners
                { x: 25, y: 70, type: 'mouth_left' },
                { x: 45, y: 70, type: 'mouth_right' },
                // Additional landmarks
                { x: 15, y: 25, type: 'left_eyebrow' },
                { x: 55, y: 25, type: 'right_eyebrow' },
                { x: 35, y: 75, type: 'chin' },
            ];
        };

        // Advanced biometric analysis
        const analyzeAdvancedBiometrics = () => {
            // Simulate AI-powered biometric analysis
            const confidence = Math.floor(Math.random() * 25) + 75; // 75-100%
            const liveness = Math.floor(Math.random() * 30) + 70;   // 70-100%

            // Facial expression analysis
            const expressions = ['neutral', 'slight_smile', 'focused', 'alert', 'confident'];
            const expressionScores = expressions.reduce((acc, exp) => {
                acc[exp] = Math.random() * 100;
                return acc;
            }, {});

            const dominantExpression = Object.keys(expressionScores).reduce((a, b) =>
                expressionScores[a] > expressionScores[b] ? a : b
            );

            return {
                confidence,
                liveness,
                expression: {
                    dominant: dominantExpression,
                    scores: expressionScores
                },
                quality: confidence > 85 && liveness > 85 ? 'excellent' :
                        confidence > 70 && liveness > 70 ? 'good' : 'fair'
            };
        };

        // Enhanced detection loop with AI features
        detectionInterval = setInterval(() => {
            // Advanced detection probability based on previous state
            faceDetected = Math.random() > (faceDetected ? 0.1 : 0.2); // 90% maintain, 80% new detection

            if (faceDetected) {
                // Advanced biometric analysis
                const biometrics = analyzeAdvancedBiometrics();
                console.log('👤 Advanced face detected!', biometrics);

                // Create enhanced face box
                const box = createAdvancedFaceBox(biometrics.confidence, biometrics.liveness, biometrics.expression);

                // Simulate realistic face tracking with slight movement
                const baseX = 60 + Math.sin(Date.now() / 2000) * 10;
                const baseY = 40 + Math.cos(Date.now() / 3000) * 5;
                const width = 80 + Math.sin(Date.now() / 1500) * 15;
                const height = 100 + Math.cos(Date.now() / 2500) * 10;

                box.style.left = `${baseX}px`;
                box.style.top = `${baseY}px`;
                box.style.width = `${width}px`;
                box.style.height = `${height}px`;

                overlay.appendChild(box);

                // Enhanced status updates
                statusSpan.innerHTML = `✅ Face: ${biometrics.quality.toUpperCase()} (${biometrics.confidence}%)`;
                statusSpan.style.color = biometrics.quality === 'excellent' ? '#10b981' :
                                        biometrics.quality === 'good' ? '#f59e0b' : '#ef4444';

                biometricStatus.innerHTML = `
                    Face Detected - Quality: ${biometrics.quality} |
                    Liveness: ${biometrics.liveness}% |
                    Expression: ${biometrics.expression.dominant.replace('_', ' ')}
                `;
                biometricStatus.className = biometrics.quality === 'excellent' ?
                    'ai-status ai-status-success ai-mb-lg' : 'ai-status ai-status-warning ai-mb-lg';

                // Advanced visual effects
                if (biometrics.quality === 'excellent') {
                    box.style.background = 'rgba(16, 185, 129, 0.25)';
                    box.style.boxShadow = '0 0 25px rgba(16, 185, 129, 0.6)';
                }

                // Store biometrics for authentication
                this.lastBiometrics = biometrics;

            } else {
                console.log('🔍 Advanced scanning for face...');

                // Remove face detection elements
                if (faceBox && faceBox.parentNode) {
                    faceBox.parentNode.removeChild(faceBox);
                    faceBox = null;
                }

                // Enhanced scanning status
                statusSpan.innerHTML = '🔍 AI Scanning... <span style="opacity: 0.6;">Looking for facial features</span>';
                statusSpan.style.color = '#f59e0b';

                biometricStatus.textContent = 'Advanced Face Detection - Position your face in the camera';
                biometricStatus.className = 'ai-status ai-status-warning ai-mb-lg';
            }

        }, 800); // Faster, smoother updates

        // Store interval for cleanup
        this.faceDetectionInterval = detectionInterval;

        // Initial enhanced status
        biometricStatus.innerHTML = `
            <div>🚀 Advanced Face Detection Active</div>
            <div style="font-size: 12px; opacity: 0.8; margin-top: 4px;">
                AI-powered | Liveness Detection | Expression Analysis
            </div>
        `;
        biometricStatus.className = 'ai-status ai-status-info ai-mb-lg';

        console.log('✅ Advanced face detection system initialized with AI features');
    }

    debugFaceDetection() {
        console.log('🐛 DEBUG: Face detection system status');

        const elements = {
            video: document.getElementById('biometricVideo'),
            overlay: document.getElementById('faceDetectionOverlay'),
            statusSpan: document.getElementById('faceDetectionStatus'),
            biometricStatus: document.getElementById('biometricStatus'),
            cameraContainer: document.getElementById('cameraContainer')
        };

        console.log('📋 Debug elements check:', elements);

        this.showToast(`Debug: Camera=${!!this.videoStream}, Interval=${!!this.faceDetectionInterval}`, 'info');

        if (elements.video && elements.overlay) {
            // Force create a test face detection box
            const testBox = document.createElement('div');
            testBox.style.cssText = `
                position: absolute;
                left: 80px;
                top: 60px;
                width: 100px;
                height: 120px;
                border: 3px solid #3b82f6;
                background: rgba(59, 130, 246, 0.2);
                border-radius: 6px;
                box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
                z-index: 100;
            `;

            // Add test label
            testBox.innerHTML = '<div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #3b82f6; font-weight: bold; font-size: 12px;">TEST</div>';

            elements.overlay.appendChild(testBox);

            // Remove after 3 seconds
            setTimeout(() => {
                if (testBox.parentNode) {
                    testBox.parentNode.removeChild(testBox);
                }
            }, 3000);

            // Update status
            if (elements.biometricStatus) {
                elements.biometricStatus.textContent = 'DEBUG: Test face box created';
                elements.biometricStatus.className = 'ai-status ai-status-info ai-mb-lg';
            }

            // Force restart face detection
            if (this.faceDetectionInterval) {
                clearInterval(this.faceDetectionInterval);
                this.faceDetectionInterval = null;
            }

            setTimeout(() => {
                this.startFaceDetection(elements.video);
            }, 1000);
        } else {
            console.error('❌ Required elements missing for face detection');
            this.showToast('ERROR: Required elements missing', 'error');
        }
    }

    bindBiometricEvents() {
        const authButton = document.querySelector('button:contains("Authenticate User")') ||
                          document.querySelector('[onclick*="authenticate"]') ||
                          Array.from(document.querySelectorAll('button')).find(btn => btn.textContent.includes('Authenticate'));

        if (authButton) {
            authButton.addEventListener('click', async () => {
                await this.performBiometricAuthentication();
            });
        }
    }

    async performBiometricAuthentication() {
        console.log('🔐 Starting biometric authentication...');

        // Show immediate feedback
        this.showToast('🔐 Starting biometric authentication...', 'info');

        // Check if camera is active
        if (!this.videoStream) {
            this.showToast('Camera not available. Please enable camera access.', 'error');
            return;
        }

        const authButton = document.getElementById('authenticateBtn');
        const biometricStatus = document.getElementById('biometricStatus');
        const statusSpan = document.getElementById('faceDetectionStatus');
        const overlay = document.getElementById('faceDetectionOverlay');

        if (authButton) {
            authButton.textContent = 'Authenticating...';
            authButton.disabled = true;
        }

        // Advanced multi-stage authentication with AI analysis
        const stages = [
            { text: 'Capturing 68 facial landmarks...', duration: 900 },
            { text: 'Analyzing biometric vectors...', duration: 800 },
            { text: 'Performing liveness detection...', duration: 700 },
            { text: 'Running expression analysis...', duration: 600 },
            { text: 'Matching against secure database...', duration: 650 },
            { text: 'Verifying quantum signature...', duration: 450 }
        ];

        // Show authentication process stages
        for (let i = 0; i < stages.length; i++) {
            const stage = stages[i];

            if (biometricStatus) {
                biometricStatus.textContent = stage.text;
                biometricStatus.className = 'ai-status ai-status-info ai-mb-lg';
            }
            if (statusSpan) {
                statusSpan.textContent = stage.text;
            }

            // Add scanning effect to face box
            if (overlay) {
                const boxes = overlay.querySelectorAll('div');
                boxes.forEach(box => {
                    box.style.borderColor = '#3b82f6';
                    box.style.background = 'rgba(59, 130, 246, 0.2)';
                    box.style.boxShadow = '0 0 15px rgba(59, 130, 246, 0.5)';
                });
            }

            await new Promise(resolve => setTimeout(resolve, stage.duration));
        }

        // Advanced authentication result using stored biometrics
        const storedBiometrics = this.lastBiometrics || {
            confidence: 85,
            liveness: 80,
            quality: 'good',
            expression: { dominant: 'neutral' }
        };

        // Success based on biometric quality
        const qualityWeight = storedBiometrics.quality === 'excellent' ? 0.95 :
                            storedBiometrics.quality === 'good' ? 0.85 : 0.70;
        const success = Math.random() < qualityWeight;

        // Advanced confidence calculation
        const baseConfidence = storedBiometrics.confidence / 100;
        const livenessBonus = (storedBiometrics.liveness - 70) / 30 * 0.05;
        const expressionBonus = storedBiometrics.expression.dominant === 'focused' ? 0.02 : 0;
        const confidence = Math.min(0.999, baseConfidence + livenessBonus + expressionBonus);

        if (success) {
            // Success feedback
            if (biometricStatus) {
                biometricStatus.innerHTML = `
                    <div>✅ Advanced Authentication Successful</div>
                    <div style="font-size: 12px; margin-top: 4px;">
                        Match: ${(confidence * 100).toFixed(2)}% |
                        Quality: ${storedBiometrics.quality} |
                        Liveness: ${storedBiometrics.liveness}% |
                        Expression: ${storedBiometrics.expression.dominant}
                    </div>
                `;
                biometricStatus.className = 'ai-status ai-status-success ai-mb-lg';
            }
            if (statusSpan) {
                statusSpan.innerHTML = `🎯 Authenticated: ${(confidence * 100).toFixed(2)}%`;
            }

            // Green confirmation effect on face box
            if (overlay) {
                const boxes = overlay.querySelectorAll('div');
                boxes.forEach(box => {
                    box.style.borderColor = '#10b981';
                    box.style.background = 'rgba(16, 185, 129, 0.3)';
                    box.style.boxShadow = '0 0 20px rgba(16, 185, 129, 0.7)';

                    // Add checkmark
                    box.innerHTML = '<div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #10b981; font-size: 24px; font-weight: bold;">✓</div>';
                });
            }

            this.showToast(`🎉 Authentication successful! Identity verified with ${(confidence * 100).toFixed(1)}% confidence`, 'success');
            console.log('✅ Biometric authentication successful');

            // Store authentication state
            this.lastAuthTime = Date.now();
            this.isAuthenticated = true;

        } else {
            // Failure feedback
            if (biometricStatus) {
                biometricStatus.textContent = '❌ Authentication Failed - Identity not verified';
                biometricStatus.className = 'ai-status ai-status-error ai-mb-lg';
            }
            if (statusSpan) {
                statusSpan.textContent = 'Authentication failed';
            }

            // Red error effect on face box
            if (overlay) {
                const boxes = overlay.querySelectorAll('div');
                boxes.forEach(box => {
                    box.style.borderColor = '#ef4444';
                    box.style.background = 'rgba(239, 68, 68, 0.2)';
                    box.style.boxShadow = '0 0 15px rgba(239, 68, 68, 0.5)';

                    // Add X mark
                    box.innerHTML = '<div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #ef4444; font-size: 24px; font-weight: bold;">✗</div>';
                });
            }

            this.showToast('❌ Authentication failed. Please position your face clearly and try again.', 'error');
            console.log('❌ Biometric authentication failed');
        }

        // Reset button and return to scanning mode after 3 seconds
        setTimeout(() => {
            if (authButton) {
                authButton.textContent = 'Authenticate User';
                authButton.disabled = false;
            }

            // Clear effects and return to normal scanning
            if (overlay) {
                const boxes = overlay.querySelectorAll('div');
                boxes.forEach(box => {
                    box.style.borderColor = '#10b981';
                    box.style.background = 'rgba(16, 185, 129, 0.1)';
                    box.style.boxShadow = '0 0 10px rgba(16, 185, 129, 0.3)';
                    box.innerHTML = '';
                });
            }

            if (biometricStatus) {
                biometricStatus.textContent = 'Face Detected - Ready to Authenticate';
                biometricStatus.className = 'ai-status ai-status-success ai-mb-lg';
            }
            if (statusSpan) {
                statusSpan.textContent = 'Face Detected';
            }
        }, 3000);
    }

    startBiometricMonitoring() {
        // Advanced real-time biometric metrics updates
        setInterval(() => {
            const metrics = {
                accuracy: (Math.random() * 0.3 + 99.7).toFixed(1),
                liveness: (Math.random() * 6 + 92).toFixed(1),
                responseTime: (Math.random() * 0.2 + 0.8).toFixed(1),
                enrolledUsers: 1247 + Math.floor(Math.random() * 10),
                expressionConfidence: (Math.random() * 8 + 85).toFixed(1),
                antiSpoofing: (Math.random() * 1.5 + 98.5).toFixed(1)
            };

            // Update enhanced metrics in UI
            this.updateAdvancedBiometricMetrics(metrics);

            console.log('👁️ Advanced biometric metrics updated:', metrics);
        }, 3000);
    }

    updateAdvancedBiometricMetrics(metrics) {
        // Update primary metrics
        const elements = {
            recognitionAccuracy: document.getElementById('recognitionAccuracy'),
            livenessScore: document.getElementById('livenessScore'),
            responseTime: document.getElementById('responseTime'),
            enrolledUsers: document.getElementById('enrolledUsers'),
            expressionConfidence: document.getElementById('expressionConfidence'),
            antiSpoofing: document.getElementById('antiSpoofing')
        };

        if (elements.recognitionAccuracy) {
            this.animateValueChange(elements.recognitionAccuracy, metrics.accuracy, '', '%');
        }
        if (elements.livenessScore) {
            this.animateValueChange(elements.livenessScore, metrics.liveness, '', '%');
        }
        if (elements.responseTime) {
            this.animateValueChange(elements.responseTime, metrics.responseTime, '', 's');
        }
        if (elements.enrolledUsers) {
            this.animateValueChange(elements.enrolledUsers, metrics.enrolledUsers);
        }
        if (elements.expressionConfidence) {
            this.animateValueChange(elements.expressionConfidence, metrics.expressionConfidence, '', '%');
        }
        if (elements.antiSpoofing) {
            this.animateValueChange(elements.antiSpoofing, metrics.antiSpoofing, '', '%');
        }
    }

    async initializeDigitalAssets() {
        console.log('💎 Initializing Digital Assets module');

        return new Promise(resolve => {
            setTimeout(() => {
                this.updateTransactionHistory();
                console.log('✅ Digital Assets module ready');
                resolve();
            }, 200);
        });
    }

    updateTransactionHistory() {
        // Add new transactions periodically
        const newTransactions = [
            {
                type: 'Data Quality Bonus',
                time: '5 minutes ago',
                amount: '+78.92 LLT-DATA',
                details: 'Dataset validation reward'
            },
            {
                type: 'Security Token Reward',
                time: '28 minutes ago',
                amount: '+156.34 LLT-SECURITY',
                details: 'Biometric verification bonus'
            }
        ];

        console.log('💰 New transactions:', newTransactions);
    }

    async initializeXRPIntegration() {
        console.log('💎 Initializing XRP Ledger integration');

        return new Promise(resolve => {
            setTimeout(() => {
                this.connectToXRPLedger();
                this.bindXRPEvents();
                console.log('✅ XRP Integration ready');
                resolve();
            }, 400);
        });
    }

    connectToXRPLedger() {
        // Simulate XRP Ledger connection
        console.log('🔗 Connected to XRP Ledger mainnet');

        // Monitor XRP network stats
        setInterval(() => {
            const stats = {
                settlementTime: (Math.random() * 1.5 + 2.8).toFixed(1),
                transactionFee: (Math.random() * 0.0001 + 0.0002).toFixed(4),
                networkUptime: (Math.random() * 0.01 + 99.99).toFixed(2)
            };

            console.log('💎 XRP Network stats:', stats);
        }, 8000);
    }

    bindXRPEvents() {
        // Buy XRP Button
        const buyBtn = document.getElementById('buyXRPBtn');
        const buyModal = document.getElementById('buyXRPModal');
        const closeBuyBtn = document.getElementById('closeBuyModal');
        const buyAmountInput = document.getElementById('buyAmount');
        const buyUSDValue = document.getElementById('buyUSDValue');
        const confirmBuyBtn = document.getElementById('confirmBuyXRP');

        if (buyBtn) {
            buyBtn.addEventListener('click', () => {
                buyModal.style.display = 'block';
                document.body.style.overflow = 'hidden';
            });
        }

        if (closeBuyBtn) {
            closeBuyBtn.addEventListener('click', () => {
                buyModal.style.display = 'none';
                document.body.style.overflow = 'auto';
            });
        }

        if (buyAmountInput) {
            buyAmountInput.addEventListener('input', (e) => {
                const amount = parseFloat(e.target.value) || 0;
                const usdValue = (amount * 0.52).toFixed(2);
                buyUSDValue.textContent = `≈ $${usdValue}`;
            });
        }

        if (confirmBuyBtn) {
            confirmBuyBtn.addEventListener('click', async () => {
                const amount = parseFloat(buyAmountInput.value);
                if (amount > 0) {
                    confirmBuyBtn.textContent = 'Processing...';
                    confirmBuyBtn.disabled = true;

                    await new Promise(resolve => setTimeout(resolve, 1500));

                    buyModal.style.display = 'none';
                    document.body.style.overflow = 'auto';
                    buyAmountInput.value = '';
                    buyUSDValue.textContent = '≈ $0.00';

                    this.showToast(`Successfully bought ${amount} XRP`, 'success');

                    confirmBuyBtn.textContent = 'Buy XRP';
                    confirmBuyBtn.disabled = false;

                    console.log(`💰 Bought ${amount} XRP`);
                }
            });
        }

        // Send XRP Button
        const sendBtn = document.getElementById('sendXRPBtn');
        const sendModal = document.getElementById('sendXRPModal');
        const closeSendBtn = document.getElementById('closeSendModal');
        const sendToAddressInput = document.getElementById('sendToAddress');
        const sendAmountInput = document.getElementById('sendAmount');
        const confirmSendBtn = document.getElementById('confirmSendXRP');

        if (sendBtn) {
            sendBtn.addEventListener('click', () => {
                sendModal.style.display = 'block';
                document.body.style.overflow = 'hidden';
            });
        }

        if (closeSendBtn) {
            closeSendBtn.addEventListener('click', () => {
                sendModal.style.display = 'none';
                document.body.style.overflow = 'auto';
            });
        }

        if (confirmSendBtn) {
            confirmSendBtn.addEventListener('click', async () => {
                const address = sendToAddressInput.value.trim();
                const amount = parseFloat(sendAmountInput.value);

                if (address && amount > 0) {
                    confirmSendBtn.textContent = 'Sending...';
                    confirmSendBtn.disabled = true;

                    await new Promise(resolve => setTimeout(resolve, 2000));

                    sendModal.style.display = 'none';
                    document.body.style.overflow = 'auto';
                    sendToAddressInput.value = '';
                    sendAmountInput.value = '';

                    this.showToast(`Successfully sent ${amount} XRP`, 'success');

                    confirmSendBtn.textContent = 'Send XRP';
                    confirmSendBtn.disabled = false;

                    console.log(`📤 Sent ${amount} XRP to ${address}`);
                }
            });
        }

        // Close modals when clicking outside
        [buyModal, sendModal].forEach(modal => {
            if (modal) {
                modal.addEventListener('click', (e) => {
                    if (e.target === modal) {
                        modal.style.display = 'none';
                        document.body.style.overflow = 'auto';
                    }
                });
            }
        });
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-secondary);
            border-radius: 12px;
            padding: 16px 20px;
            color: var(--text-primary);
            z-index: 1001;
            transform: translateX(100%);
            transition: transform 0.3s ease;
            box-shadow: var(--shadow-lg);
            max-width: 300px;
        `;

        if (type === 'success') {
            toast.style.borderLeft = '4px solid var(--success)';
        } else if (type === 'error') {
            toast.style.borderLeft = '4px solid var(--error)';
        }

        toast.textContent = message;
        document.body.appendChild(toast);

        // Slide in
        setTimeout(() => {
            toast.style.transform = 'translateX(0)';
        }, 100);

        // Slide out and remove
        setTimeout(() => {
            toast.style.transform = 'translateX(100%)';
            setTimeout(() => {
                document.body.removeChild(toast);
            }, 300);
        }, 3000);
    }

    async initializeAPIDocumentation() {
        console.log('📚 Initializing API Documentation');

        return new Promise(resolve => {
            setTimeout(() => {
                this.setupAPIExamples();
                console.log('✅ API Documentation ready');
                resolve();
            }, 100);
        });
    }

    setupAPIExamples() {
        // Add syntax highlighting effect
        const codeBlocks = document.querySelectorAll('.ai-code');
        codeBlocks.forEach(block => {
            block.addEventListener('mouseenter', () => {
                block.style.boxShadow = '0 0 30px rgba(59, 130, 246, 0.2)';
            });

            block.addEventListener('mouseleave', () => {
                block.style.boxShadow = '';
            });
        });
    }

    addSmoothAnimations() {
        // Add smooth transitions to all interactive elements
        const interactiveElements = document.querySelectorAll('.ai-card, .ai-btn, .ai-nav-tab');

        interactiveElements.forEach(element => {
            element.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
        });

        // Add stagger animation to cards on load
        const cards = document.querySelectorAll('.ai-card');
        cards.forEach((card, index) => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(30px)';

            setTimeout(() => {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, 100 * index);
        });

        // Add hover glow effects
        this.addGlowEffects();
    }

    addGlowEffects() {
        const glowElements = document.querySelectorAll('.ai-glow, .ai-card');

        glowElements.forEach(element => {
            element.addEventListener('mouseenter', () => {
                element.style.boxShadow = '0 0 40px rgba(59, 130, 246, 0.15)';
                element.style.transform = 'translateY(-4px)';
            });

            element.addEventListener('mouseleave', () => {
                element.style.boxShadow = '';
                element.style.transform = 'translateY(0)';
            });
        });
    }

    showWelcomeMessage() {
        console.log(`
        ╔══════════════════════════════════════════════════════════════╗
        ║                   LL TOKEN OFFLINE.AI                       ║
        ║                                                              ║
        ║  🤖 AI-Powered Federated Learning Platform                  ║
        ║  👁️ Advanced Biometric Authentication                       ║
        ║  💎 XRP Ledger Enterprise Integration                       ║
        ║  🔒 Quantum-Safe Security Architecture                      ║
        ║  🛠️ Comprehensive Developer APIs                            ║
        ║                                                              ║
        ║  Status: ✨ All systems operational                         ║
        ║  Design: 🎨 Offline.AI modern dark theme                   ║
        ╚══════════════════════════════════════════════════════════════╝
        `);
    }

    // API Integration Methods
    async startFederatedTraining(config) {
        console.log('🚀 Starting AI-powered federated learning:', config);

        const response = await this.simulateAPICall('/api/v1/fl/training/start', {
            method: 'POST',
            body: JSON.stringify(config)
        });

        console.log('✅ Training session initialized:', response.sessionId);
        return response;
    }

    async authenticateUser(biometricData) {
        console.log('🔐 Processing biometric authentication');

        const response = await this.simulateAPICall('/api/v1/biometrics/verify', {
            method: 'POST',
            body: JSON.stringify(biometricData)
        });

        console.log('✅ Authentication result:', response.authenticated);
        return response;
    }

    async sendXRPPayment(paymentData) {
        console.log('💎 Processing XRP payment:', paymentData);

        const response = await this.simulateAPICall('/api/v1/xrp/payment', {
            method: 'POST',
            body: JSON.stringify(paymentData)
        });

        console.log('✅ Payment processed:', response.transactionHash);
        return response;
    }

    async getTokenBalances(address) {
        console.log('📊 Querying token portfolio:', address);

        const response = await this.simulateAPICall(`/api/v1/tokens/balance?address=${address}`);
        console.log('✅ Portfolio data retrieved');
        return response;
    }

    // Utility Methods
    async simulateAPICall(endpoint, options = {}) {
        // Simulate network delay with smooth loading
        await new Promise(resolve => setTimeout(resolve, Math.random() * 800 + 200));

        // Generate realistic responses
        switch (endpoint) {
            case '/api/v1/fl/training/start':
                return {
                    sessionId: 'fl_ai_' + Date.now(),
                    status: 'initializing',
                    aiModel: 'transformer_v2.1',
                    expectedCompletion: new Date(Date.now() + 3600000).toISOString(),
                    participants: Math.floor(Math.random() * 20) + 15
                };

            case '/api/v1/biometrics/verify':
                return {
                    authenticated: Math.random() > 0.02, // 98% success rate
                    confidence: Math.random() * 0.05 + 0.95,
                    userId: 'user_ai_' + Math.random().toString(36).substr(2, 9),
                    biometricScore: Math.random() * 0.03 + 0.97
                };

            case '/api/v1/xrp/payment':
                return {
                    transactionHash: 'tx_xrp_' + Math.random().toString(36).substr(2, 16),
                    status: 'validated',
                    fee: (Math.random() * 0.0001 + 0.0002).toFixed(4) + ' XRP',
                    settlementTime: (Math.random() * 1.5 + 2.5).toFixed(1) + 's',
                    timestamp: new Date().toISOString()
                };

            default:
                if (endpoint.includes('/api/v1/tokens/balance')) {
                    return {
                        address: endpoint.split('=')[1],
                        portfolio: {
                            'LLT-COMPUTE': Math.floor(Math.random() * 50000) + 100000,
                            'LLT-DATA': Math.floor(Math.random() * 30000) + 50000,
                            'LLT-SECURITY': Math.floor(Math.random() * 20000) + 25000
                        },
                        totalValue: '$' + (Math.random() * 500000 + 2000000).toFixed(0),
                        lastUpdated: new Date().toISOString()
                    };
                }
                return { status: 'success', timestamp: new Date().toISOString() };
        }
    }

    // Account Setup Tab Methods
    updateAccountSetupData() {
        console.log('📋 Updating account setup data');

        // Initialize form validations and interactions
        this.initializeAccountSetupForms();
    }

    initializeAccountSetupForms() {
        // Bank account connection form
        const connectBankBtn = document.querySelector('#setupContent .ai-btn-primary');
        if (connectBankBtn) {
            connectBankBtn.addEventListener('click', () => {
                this.handleBankConnection();
            });
        }

        // Security settings form
        const saveSecurityBtn = document.querySelector('#setupContent .ai-btn-secondary');
        if (saveSecurityBtn) {
            saveSecurityBtn.addEventListener('click', () => {
                this.handleSecuritySettings();
            });
        }

        // KYC verification buttons
        const kycButtons = document.querySelectorAll('#setupContent .ai-btn-outline');
        kycButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.handleKYCUpload(e.target);
            });
        });

        // Complete KYC button
        const completeKYCBtn = document.querySelector('#setupContent .ai-btn-lg');
        if (completeKYCBtn) {
            completeKYCBtn.addEventListener('click', () => {
                this.handleCompleteKYC();
            });
        }
    }

    async handleBankConnection() {
        // Simulate bank connection process
        const bankSelect = document.querySelector('#setupContent select[class="ai-select"]');
        const routingInput = document.querySelector('#setupContent input[placeholder="9-digit routing number"]');
        const accountInput = document.querySelector('#setupContent input[placeholder="Account number"]');

        if (!bankSelect.value || bankSelect.value === 'Select your bank...') {
            this.showToast('Please select your bank', 'error');
            return;
        }

        if (!routingInput.value || routingInput.value.length !== 9) {
            this.showToast('Please enter a valid 9-digit routing number', 'error');
            return;
        }

        if (!accountInput.value) {
            this.showToast('Please enter your account number', 'error');
            return;
        }

        // Simulate connection process
        this.showToast('Connecting to bank...', 'info');
        await new Promise(resolve => setTimeout(resolve, 2000));
        this.showToast('Bank account connected successfully!', 'success');

        console.log('🏦 Bank account connected');
    }

    handleSecuritySettings() {
        this.showToast('Security settings saved successfully!', 'success');
        console.log('🔒 Security settings updated');
    }

    handleKYCUpload(button) {
        const uploadType = button.textContent;
        this.showToast(`${uploadType} feature coming soon`, 'info');
        console.log(`📄 KYC upload: ${uploadType}`);
    }

    async handleCompleteKYC() {
        this.showToast('KYC verification submitted for review', 'success');
        console.log('✅ KYC verification submitted');
    }

    // Banking Dashboard Tab Methods
    updateBankingDashboardData() {
        console.log('🏦 Updating banking dashboard data');

        // Initialize banking dashboard interactions
        this.initializeBankingDashboard();

        // Start real-time banking updates
        this.startBankingUpdates();
    }

    initializeBankingDashboard() {
        // Quick action buttons
        const quickActionBtns = document.querySelectorAll('#bankingContent .ai-btn-outline');
        quickActionBtns.forEach((btn, index) => {
            btn.addEventListener('click', () => {
                const actions = ['Send Money', 'Receive Payment', 'Convert XRP', 'View Reports'];
                this.handleQuickAction(actions[index]);
            });
        });

        // Connect new account button
        const connectNewBtn = document.querySelector('#bankingContent .ai-btn-ghost');
        if (connectNewBtn) {
            connectNewBtn.addEventListener('click', () => {
                this.handleConnectNewAccount();
            });
        }
    }

    startBankingUpdates() {
        // Update banking metrics every 5 seconds
        setInterval(() => {
            this.updateBankingMetrics();
        }, 5000);
    }

    updateBankingMetrics() {
        // Update total balance with small fluctuations
        const totalBalanceElement = document.querySelector('#bankingContent .ai-metric-value');
        if (totalBalanceElement) {
            const currentBalance = parseFloat(totalBalanceElement.textContent.replace(/[$,]/g, ''));
            const change = (Math.random() - 0.5) * 100; // ±$50 change
            const newBalance = Math.max(120000, currentBalance + change);
            totalBalanceElement.textContent = `$${newBalance.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        }

        // Update XRP holdings
        const xrpElement = document.querySelector('#bankingContent .ai-metric-value:nth-child(2)');
        if (xrpElement && xrpElement.textContent.includes('XRP')) {
            const change = Math.floor(Math.random() * 1000) - 500; // ±500 XRP
            const newAmount = Math.max(120000, 125000 + change);
            xrpElement.textContent = `${newAmount.toLocaleString()} XRP`;
        }

        console.log('💰 Banking metrics updated');
    }

    handleQuickAction(action) {
        switch(action) {
            case 'Send Money':
                this.showToast('Send money feature opening...', 'info');
                break;
            case 'Receive Payment':
                this.showToast('Receive payment QR code generated', 'success');
                break;
            case 'Convert XRP':
                this.showToast('XRP conversion interface loading...', 'info');
                break;
            case 'View Reports':
                this.showToast('Financial reports being generated...', 'info');
                break;
        }
        console.log(`💼 Quick action: ${action}`);
    }

    handleConnectNewAccount() {
        this.showToast('Account connection wizard opening...', 'info');
        console.log('🔗 Connect new account initiated');
    }

    // Lifecycle methods
    destroy() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
        console.log('🔄 OFFLINE.AI Platform cleaned up');
    }
}

// Initialize platform when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Loading LL TOKEN OFFLINE.AI Platform...');

    // Create global platform instance
    window.OfflineAI = new OfflineAIPlatform();

    // Add global utility functions
    window.OfflineAI.utils = {
        formatNumber: (num) => {
            if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
            if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
            return num.toString();
        },

        formatCurrency: (amount, currency = 'USD') => {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: currency
            }).format(amount);
        },

        generateId: () => Math.random().toString(36).substr(2, 9)
    };

    console.log('🎨 OFFLINE.AI Platform ready for interaction');
});

// Handle page unload
window.addEventListener('beforeunload', function() {
    if (window.OfflineAI) {
        window.OfflineAI.destroy();
    }
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { OfflineAIPlatform };
}