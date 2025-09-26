/**
 * Enhanced Widgets - Upper-Level Graphics for FL Dashboard
 * Advanced visual components with modern animations and interactions
 */

class EnhancedFLWidgets {
    constructor() {
        this.animationFrameId = null;
        this.chartData = [];
        this.networkNodes = [];
        this.connections = [];
        this.initialized = false;
    }

    // Initialize all enhanced widgets
    init() {
        if (this.initialized) return;

        this.createNetworkVisualization();
        this.initializeCharts();
        this.setupAnimations();
        this.setupInteractions();

        this.initialized = true;
        console.log('🎨 Enhanced widgets initialized');
    }

    // Create advanced network visualization
    createNetworkVisualization() {
        const networkContainer = document.querySelector('.network-viz');
        if (!networkContainer) return;

        // Create SVG overlay for connections
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.style.position = 'absolute';
        svg.style.top = '0';
        svg.style.left = '0';
        svg.style.width = '100%';
        svg.style.height = '100%';
        svg.style.pointerEvents = 'none';
        svg.style.zIndex = '1';

        networkContainer.appendChild(svg);

        // Create animated connection lines
        this.createConnectionLines(svg);

        // Add particle effects
        this.createParticleEffects(networkContainer);
    }

    // Create animated connection lines between nodes
    createConnectionLines(svg) {
        const nodes = document.querySelectorAll('.node');
        const serverNode = document.querySelector('.node.server');

        if (!serverNode) return;

        const serverRect = serverNode.getBoundingClientRect();
        const containerRect = svg.getBoundingClientRect();

        nodes.forEach(node => {
            if (node === serverNode) return;

            const nodeRect = node.getBoundingClientRect();

            // Create animated path
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');

            const startX = serverRect.left - containerRect.left + serverRect.width / 2;
            const startY = serverRect.top - containerRect.top + serverRect.height / 2;
            const endX = nodeRect.left - containerRect.left + nodeRect.width / 2;
            const endY = nodeRect.top - containerRect.top + nodeRect.height / 2;

            // Create curved path
            const midX = (startX + endX) / 2;
            const midY = Math.min(startY, endY) - 50;

            const pathData = `M ${startX} ${startY} Q ${midX} ${midY} ${endX} ${endY}`;

            path.setAttribute('d', pathData);
            path.setAttribute('stroke', '#10a37f');
            path.setAttribute('stroke-width', '2');
            path.setAttribute('fill', 'none');
            path.setAttribute('opacity', '0.6');
            path.style.filter = 'drop-shadow(0 0 5px #10a37f)';

            // Add animation
            const pathLength = path.getTotalLength();
            path.style.strokeDasharray = pathLength;
            path.style.strokeDashoffset = pathLength;
            path.style.animation = `drawPath 3s ease-in-out infinite`;

            svg.appendChild(path);
        });

        // Add CSS animation for path drawing
        const style = document.createElement('style');
        style.textContent = `
            @keyframes drawPath {
                0% { stroke-dashoffset: ${path?.getTotalLength() || 0}; opacity: 0.3; }
                50% { opacity: 1; }
                100% { stroke-dashoffset: 0; opacity: 0.3; }
            }
        `;
        document.head.appendChild(style);
    }

    // Create particle effects for data flow
    createParticleEffects(container) {
        const particleContainer = document.createElement('div');
        particleContainer.className = 'particle-container';
        particleContainer.style.position = 'absolute';
        particleContainer.style.top = '0';
        particleContainer.style.left = '0';
        particleContainer.style.width = '100%';
        particleContainer.style.height = '100%';
        particleContainer.style.pointerEvents = 'none';
        particleContainer.style.zIndex = '2';

        container.appendChild(particleContainer);

        // Create floating particles
        for (let i = 0; i < 20; i++) {
            this.createParticle(particleContainer);
        }
    }

    createParticle(container) {
        const particle = document.createElement('div');
        particle.className = 'data-particle';
        particle.style.position = 'absolute';
        particle.style.width = '4px';
        particle.style.height = '4px';
        particle.style.background = '#10a37f';
        particle.style.borderRadius = '50%';
        particle.style.boxShadow = '0 0 10px #10a37f';

        // Random starting position
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = Math.random() * 100 + '%';

        // Random animation
        const duration = 3 + Math.random() * 4;
        particle.style.animation = `floatParticle ${duration}s linear infinite`;
        particle.style.animationDelay = Math.random() * 2 + 's';

        container.appendChild(particle);

        // Add CSS animation for particles
        if (!document.getElementById('particle-styles')) {
            const style = document.createElement('style');
            style.id = 'particle-styles';
            style.textContent = `
                @keyframes floatParticle {
                    0% {
                        transform: translate(0, 0) scale(0);
                        opacity: 0;
                    }
                    10% {
                        opacity: 1;
                        transform: scale(1);
                    }
                    90% {
                        opacity: 1;
                    }
                    100% {
                        transform: translate(${(Math.random() - 0.5) * 200}px, ${(Math.random() - 0.5) * 200}px) scale(0);
                        opacity: 0;
                    }
                }
            `;
            document.head.appendChild(style);
        }
    }

    // Initialize advanced charts
    initializeCharts() {
        this.createRealtimeChart();
        this.createNetworkTopologyChart();
        this.createPerformanceMetrics();
    }

    // Create real-time training progress chart
    createRealtimeChart() {
        const chartContainer = document.querySelector('.chart-container');
        if (!chartContainer) return;

        // Clear existing content
        chartContainer.innerHTML = '';

        // Create SVG chart
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.style.width = '100%';
        svg.style.height = '100%';

        // Create gradient
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
        gradient.id = 'chartGradient';
        gradient.setAttribute('x1', '0%');
        gradient.setAttribute('y1', '0%');
        gradient.setAttribute('x2', '100%');
        gradient.setAttribute('y2', '0%');

        const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        stop1.setAttribute('offset', '0%');
        stop1.setAttribute('stop-color', '#10a37f');

        const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        stop2.setAttribute('offset', '100%');
        stop2.setAttribute('stop-color', '#1a73e8');

        gradient.appendChild(stop1);
        gradient.appendChild(stop2);
        defs.appendChild(gradient);
        svg.appendChild(defs);

        // Create animated path
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('stroke', 'url(#chartGradient)');
        path.setAttribute('stroke-width', '3');
        path.setAttribute('fill', 'none');
        path.style.filter = 'drop-shadow(0 0 10px rgba(16, 163, 127, 0.5))';

        svg.appendChild(path);
        chartContainer.appendChild(svg);

        // Animate the chart
        this.animateChart(path);
    }

    // Animate chart with real-time data
    animateChart(path) {
        let points = [];
        const maxPoints = 50;
        let currentAccuracy = 85;

        const updateChart = () => {
            // Simulate data points
            currentAccuracy += (Math.random() - 0.5) * 2;
            currentAccuracy = Math.max(80, Math.min(99, currentAccuracy));

            points.push(currentAccuracy);
            if (points.length > maxPoints) {
                points.shift();
            }

            // Create path data
            const width = 300; // Approximate width
            const height = 150; // Approximate height
            const stepX = width / (maxPoints - 1);

            let pathData = '';
            points.forEach((point, index) => {
                const x = index * stepX;
                const y = height - ((point - 80) / 19) * height; // Normalize to 80-99 range

                if (index === 0) {
                    pathData += `M ${x} ${y}`;
                } else {
                    pathData += ` L ${x} ${y}`;
                }
            });

            path.setAttribute('d', pathData);

            // Continue animation
            setTimeout(updateChart, 2000);
        };

        updateChart();
    }

    // Create network topology visualization
    createNetworkTopologyChart() {
        const nodes = document.querySelectorAll('.node');

        nodes.forEach(node => {
            // Add hover effects
            node.addEventListener('mouseenter', () => {
                node.style.transform = 'scale(1.1)';
                node.style.transition = 'all 0.3s ease';
                node.style.boxShadow = '0 0 20px rgba(16, 163, 127, 0.8)';
            });

            node.addEventListener('mouseleave', () => {
                node.style.transform = 'scale(1)';
                node.style.boxShadow = '';
            });

            // Add click effects
            node.addEventListener('click', () => {
                this.showNodeDetails(node);
            });
        });
    }

    // Show detailed node information
    showNodeDetails(node) {
        const modal = document.createElement('div');
        modal.className = 'node-modal';
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        `;

        const content = document.createElement('div');
        content.style.cssText = `
            background: var(--bg-card);
            border: 2px solid var(--accent-primary);
            border-radius: 16px;
            padding: 30px;
            max-width: 400px;
            width: 90%;
            color: white;
        `;

        const nodeType = node.classList.contains('server') ? 'Server' : 'Client';
        const status = node.classList.contains('active') ? 'Active' : 'Inactive';

        content.innerHTML = `
            <h3 style="color: var(--accent-primary); margin-bottom: 20px;">
                ${nodeType} Details
            </h3>
            <div style="margin-bottom: 15px;">
                <strong>Status:</strong> ${status}
            </div>
            <div style="margin-bottom: 15px;">
                <strong>Type:</strong> ${nodeType}
            </div>
            <div style="margin-bottom: 15px;">
                <strong>Performance:</strong> ${85 + Math.random() * 10}%
            </div>
            <div style="margin-bottom: 20px;">
                <strong>Data Points:</strong> ${Math.floor(Math.random() * 10000) + 5000}
            </div>
            <button onclick="this.parentElement.parentElement.remove()"
                    style="background: var(--accent-primary); color: white; border: none;
                           padding: 10px 20px; border-radius: 8px; cursor: pointer;">
                Close
            </button>
        `;

        modal.appendChild(content);
        document.body.appendChild(modal);

        // Close on outside click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }

    // Create performance metrics visualization
    createPerformanceMetrics() {
        const metricCards = document.querySelectorAll('.metric-card');

        metricCards.forEach(card => {
            // Add animated background
            card.style.position = 'relative';
            card.style.overflow = 'hidden';

            const shimmer = document.createElement('div');
            shimmer.style.cssText = `
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(16, 163, 127, 0.1), transparent);
                animation: shimmer 3s infinite;
                pointer-events: none;
            `;

            card.appendChild(shimmer);

            // Add CSS animation
            if (!document.getElementById('shimmer-animation')) {
                const style = document.createElement('style');
                style.id = 'shimmer-animation';
                style.textContent = `
                    @keyframes shimmer {
                        0% { left: -100%; }
                        100% { left: 100%; }
                    }
                `;
                document.head.appendChild(style);
            }
        });
    }

    // Setup advanced animations
    setupAnimations() {
        // Add scroll-triggered animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.animation = 'fadeInUp 0.8s ease-out';
                }
            });
        }, observerOptions);

        // Observe all cards
        document.querySelectorAll('.card, .metric-card, .action-card').forEach(card => {
            observer.observe(card);
        });

        // Add fade-in animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
        `;
        document.head.appendChild(style);
    }

    // Setup interactive elements
    setupInteractions() {
        // Add advanced button effects
        document.querySelectorAll('.control-btn, .vault-btn, .btn').forEach(button => {
            button.addEventListener('mouseenter', () => {
                button.style.transform = 'translateY(-2px) scale(1.02)';
                button.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
            });

            button.addEventListener('mouseleave', () => {
                button.style.transform = 'translateY(0) scale(1)';
            });
        });

        // Add ripple effect to clickable elements
        this.addRippleEffect();
    }

    // Add ripple effect to buttons
    addRippleEffect() {
        document.querySelectorAll('.control-btn, .vault-btn, .btn').forEach(button => {
            button.addEventListener('click', (e) => {
                const ripple = document.createElement('span');
                const rect = button.getBoundingClientRect();
                const size = Math.max(rect.width, rect.height);
                const x = e.clientX - rect.left - size / 2;
                const y = e.clientY - rect.top - size / 2;

                ripple.style.cssText = `
                    position: absolute;
                    width: ${size}px;
                    height: ${size}px;
                    left: ${x}px;
                    top: ${y}px;
                    background: rgba(255, 255, 255, 0.3);
                    border-radius: 50%;
                    pointer-events: none;
                    animation: ripple 0.6s ease-out;
                `;

                button.style.position = 'relative';
                button.style.overflow = 'hidden';
                button.appendChild(ripple);

                setTimeout(() => ripple.remove(), 600);
            });
        });

        // Add ripple animation
        if (!document.getElementById('ripple-animation')) {
            const style = document.createElement('style');
            style.id = 'ripple-animation';
            style.textContent = `
                @keyframes ripple {
                    0% {
                        transform: scale(0);
                        opacity: 1;
                    }
                    100% {
                        transform: scale(2);
                        opacity: 0;
                    }
                }
            `;
            document.head.appendChild(style);
        }
    }

    // Update metrics with smooth animations
    updateMetric(elementId, newValue, suffix = '') {
        const element = document.getElementById(elementId);
        if (!element) return;

        const currentValue = parseFloat(element.textContent) || 0;
        const targetValue = parseFloat(newValue) || 0;
        const duration = 1000; // 1 second
        const steps = 60; // 60 FPS
        const increment = (targetValue - currentValue) / steps;
        let current = currentValue;
        let step = 0;

        const animate = () => {
            if (step < steps) {
                current += increment;
                element.textContent = current.toFixed(1) + suffix;
                step++;
                requestAnimationFrame(animate);
            } else {
                element.textContent = targetValue.toFixed(1) + suffix;
            }
        };

        animate();
    }

    // Destroy all widgets and clean up
    destroy() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }

        // Remove created elements
        document.querySelectorAll('.particle-container, .node-modal').forEach(el => {
            el.remove();
        });

        this.initialized = false;
        console.log('🧹 Enhanced widgets destroyed');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EnhancedFLWidgets;
} else {
    window.EnhancedFLWidgets = EnhancedFLWidgets;
}

// Auto-initialize when DOM is ready
if (typeof document !== 'undefined') {
    document.addEventListener('DOMContentLoaded', () => {
        window.flWidgets = new EnhancedFLWidgets();
        window.flWidgets.init();
    });
}