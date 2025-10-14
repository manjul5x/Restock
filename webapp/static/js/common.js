// Common JavaScript functions for Forecaster webapp

// Format number with thousands separator and decimal places
function formatNumber(num, decimals = 0) {
    if (num === null || num === undefined) return '-';
    if (typeof num === 'string') return num;
    
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    } else {
        return num.toFixed(decimals);
    }
}

// Format percentage
function formatPercentage(num) {
    if (num === null || num === undefined) return '-';
    return (num * 100).toFixed(2) + '%';
}

// Format currency
function formatCurrency(num) {
    if (num === null || num === undefined) return '-';
    if (num >= 1000000) {
        return '$' + (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return '$' + (num / 1000).toFixed(1) + 'K';
    } else {
        return '$' + num.toFixed(2);
    }
}

// Format date
function formatDate(date) {
    if (!(date instanceof Date)) {
        date = new Date(date);
    }
    return date.toISOString().split('T')[0];
}

// Show loading spinner
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.style.display = 'block';
    }
}

// Hide loading spinner
function hideLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.style.display = 'none';
    }
}

// Show error message
function showError(message, elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = message;
        element.style.display = 'block';
    } else {
        console.error(message);
    }
}

// Hide error message
function hideError(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.style.display = 'none';
    }
}

// Debounce function for rate limiting
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Handle AJAX errors
function handleAjaxError(error, elementId) {
    console.error('AJAX Error:', error);
    showError('An error occurred while processing your request. Please try again.', elementId);
}

// Update Plotly layout
function updatePlotlyLayout(layout, title) {
    return {
        ...layout,
        title: {
            text: title,
            font: {
                family: 'Segoe UI, sans-serif',
                size: 24
            }
        },
        paper_bgcolor: 'rgba(255,255,255,0.95)',
        plot_bgcolor: 'rgba(255,255,255,0.95)',
        font: {
            family: 'Segoe UI, sans-serif'
        },
        margin: {
            l: 50,
            r: 50,
            t: 80,
            b: 50
        },
        showlegend: true,
        legend: {
            orientation: 'h',
            yanchor: 'bottom',
            y: -0.2,
            xanchor: 'center',
            x: 0.5
        },
        xaxis: {
            showgrid: true,
            gridcolor: '#e9ecef',
            tickfont: {
                family: 'Segoe UI, sans-serif'
            }
        },
        yaxis: {
            showgrid: true,
            gridcolor: '#e9ecef',
            tickfont: {
                family: 'Segoe UI, sans-serif'
            }
        }
    };
} 