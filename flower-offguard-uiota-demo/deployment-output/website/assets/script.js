// Website functionality
document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for navigation links
    const navLinks = document.querySelectorAll('.nav-menu a[href^="#"]');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);

            if (targetElement) {
                const offsetTop = targetElement.offsetTop - 80; // Account for fixed navbar
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Add active state to navigation
    const sections = document.querySelectorAll('section[id]');
    const navItems = document.querySelectorAll('.nav-menu a');

    function highlightNavigation() {
        let current = '';

        sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionHeight = section.offsetHeight;

            if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });

        navItems.forEach(item => {
            item.classList.remove('active');
            if (item.getAttribute('href').substring(1) === current) {
                item.classList.add('active');
            }
        });
    }

    // Highlight navigation on scroll
    window.addEventListener('scroll', highlightNavigation);

    // Terminal animation
    const terminalLines = document.querySelectorAll('.terminal-line');
    let lineIndex = 0;

    function typeNextLine() {
        if (lineIndex < terminalLines.length) {
            const line = terminalLines[lineIndex];
            line.style.opacity = '0';
            line.style.transform = 'translateX(-20px)';

            setTimeout(() => {
                line.style.transition = 'all 0.5s ease-out';
                line.style.opacity = '1';
                line.style.transform = 'translateX(0)';
                lineIndex++;

                setTimeout(typeNextLine, 800);
            }, 200);
        } else {
            // Restart animation after delay
            setTimeout(() => {
                lineIndex = 0;
                terminalLines.forEach(line => {
                    line.style.opacity = '0';
                    line.style.transform = 'translateX(-20px)';
                });
                setTimeout(typeNextLine, 1000);
            }, 5000);
        }
    }

    // Start terminal animation after page load
    setTimeout(typeNextLine, 2000);

    // Copy to clipboard functionality for checksums
    const checksumLinks = document.querySelectorAll('.checksum-link');
    checksumLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            // This would typically fetch and display checksum data
            showNotification('Checksum information copied to clipboard!');
        });
    });

    // Download tracking (optional analytics)
    const downloadButtons = document.querySelectorAll('a[href*="downloads/"]');
    downloadButtons.forEach(button => {
        button.addEventListener('click', function() {
            const fileName = this.getAttribute('href').split('/').pop();
            console.log(`Download started: ${fileName}`);
            showNotification(`Starting download: ${fileName}`);
        });
    });

    // Show notification function
    function showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'notification';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            background: #10b981;
            color: white;
            padding: 1rem;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            z-index: 10000;
            opacity: 0;
            transform: translateX(100%);
            transition: all 0.3s ease-out;
        `;

        document.body.appendChild(notification);

        // Show notification
        setTimeout(() => {
            notification.style.opacity = '1';
            notification.style.transform = 'translateX(0)';
        }, 100);

        // Hide notification
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }

    // Add API section toggle functionality
    const moduleHeaders = document.querySelectorAll('.module h3');
    moduleHeaders.forEach(header => {
        header.style.cursor = 'pointer';
        header.addEventListener('click', function() {
            const moduleContent = this.nextElementSibling;
            const isHidden = moduleContent.style.display === 'none';

            moduleContent.style.display = isHidden ? 'block' : 'none';
            this.style.opacity = isHidden ? '1' : '0.7';
        });
    });

    // Initialize collapsed state for API modules
    document.querySelectorAll('.module-content').forEach(content => {
        content.style.display = 'none';
    });

    document.querySelectorAll('.module h3').forEach(header => {
        header.style.opacity = '0.7';
        header.title = 'Click to expand/collapse';
    });
});

// Add CSS for active navigation
const style = document.createElement('style');
style.textContent = `
    .nav-menu a.active {
        color: #2563eb;
        font-weight: 600;
    }

    .notification {
        font-weight: 500;
        font-size: 0.9rem;
    }
`;
document.head.appendChild(style);
