/**
 * Off-Guard API Client
 * Handles all backend communication with authentication
 */

class OffGuardAPI {
    constructor() {
        this.baseURL = 'http://localhost:8002/api';
        this.token = localStorage.getItem('auth_token');
    }

    // Authentication helpers
    getAuthHeaders() {
        const headers = {
            'Content-Type': 'application/json'
        };

        if (this.token) {
            headers['Authorization'] = `Bearer ${this.token}`;
        }

        return headers;
    }

    async makeRequest(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: this.getAuthHeaders(),
            ...options
        };

        try {
            const response = await fetch(url, config);
            const data = await response.json();

            if (response.status === 401) {
                // Token expired or invalid
                this.logout();
                window.location.href = '/auth.html';
                return null;
            }

            return { success: response.ok, data, status: response.status };
        } catch (error) {
            console.error('API Request failed:', error);
            return { success: false, error: error.message };
        }
    }

    // Authentication methods
    async login(username, password) {
        const result = await this.makeRequest('/auth/login', {
            method: 'POST',
            body: JSON.stringify({ username, password })
        });

        if (result.success && result.data.token) {
            this.token = result.data.token;
            localStorage.setItem('auth_token', this.token);
            localStorage.setItem('user_data', JSON.stringify(result.data.user));
        }

        return result;
    }

    async signup(username, email, password) {
        const result = await this.makeRequest('/auth/signup', {
            method: 'POST',
            body: JSON.stringify({ username, email, password })
        });

        if (result.success && result.data.token) {
            this.token = result.data.token;
            localStorage.setItem('auth_token', this.token);
            localStorage.setItem('user_data', JSON.stringify(result.data.user));
        }

        return result;
    }

    logout() {
        this.token = null;
        localStorage.removeItem('auth_token');
        localStorage.removeItem('user_data');
    }

    // Federated Learning endpoints
    async startFLDemo() {
        return await this.makeRequest('/fl/start-demo', {
            method: 'POST'
        });
    }

    async addFLClient(name = null, type = 'mobile') {
        return await this.makeRequest('/fl/add-client', {
            method: 'POST',
            body: JSON.stringify({ name, type })
        });
    }

    async runTraining() {
        return await this.makeRequest('/fl/run-training', {
            method: 'POST'
        });
    }

    // AI Integration endpoints
    async testAI(provider) {
        return await this.makeRequest('/ai/test', {
            method: 'POST',
            body: JSON.stringify({ provider })
        });
    }

    async analyzeModel() {
        return await this.makeRequest('/ai/analyze', {
            method: 'POST'
        });
    }

    // Security endpoints
    async generateKeys() {
        return await this.makeRequest('/security/generate-keys', {
            method: 'POST'
        });
    }

    async testEncryption() {
        return await this.makeRequest('/security/test-encryption', {
            method: 'POST'
        });
    }

    async enableOfflineMode() {
        return await this.makeRequest('/security/offline-mode', {
            method: 'POST'
        });
    }

    // API Management
    async saveApiKeys(keys) {
        return await this.makeRequest('/keys/save', {
            method: 'POST',
            body: JSON.stringify(keys)
        });
    }

    // Metrics and Data
    async getMetrics() {
        return await this.makeRequest('/metrics');
    }

    async exportData() {
        return await this.makeRequest('/export');
    }

    // Utility methods
    isAuthenticated() {
        return !!this.token;
    }

    getCurrentUser() {
        const userData = localStorage.getItem('user_data');
        return userData ? JSON.parse(userData) : null;
    }

    // Frontend notification helpers
    showNotification(message, type = 'info') {
        // Create notification element if it doesn't exist
        let container = document.getElementById('notification-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'notification-container';
            container.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10000;
                max-width: 400px;
            `;
            document.body.appendChild(container);
        }

        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.style.cssText = `
            background: ${type === 'success' ? '#22c55e' : type === 'error' ? '#ef4444' : '#3b82f6'};
            color: white;
            padding: 15px 20px;
            margin-bottom: 10px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            animation: slideIn 0.3s ease-out;
            cursor: pointer;
        `;

        notification.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <span>${type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️'}</span>
                <span>${message}</span>
            </div>
        `;

        notification.onclick = () => notification.remove();

        container.appendChild(notification);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }

    // Authentication check for protected routes
    requireAuth() {
        if (!this.isAuthenticated()) {
            window.location.href = '/auth.html';
            return false;
        }
        return true;
    }
}

// Global API instance
window.api = new OffGuardAPI();

// Add styles for notifications
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    .notification {
        transition: all 0.3s ease;
    }

    .notification:hover {
        transform: translateX(-5px);
    }
`;
document.head.appendChild(notificationStyles);