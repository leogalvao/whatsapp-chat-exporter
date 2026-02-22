// WhatsApp Chat Exporter - Background Service Worker
// Handles downloads and other background tasks

// Listen for messages from content script and popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'download') {
    handleDownload(message.url, message.filename);
    sendResponse({ success: true });
  }
  
  // progressUpdate messages are sent by the content script directly to all
  // extension views (including the popup) via chrome.runtime.sendMessage.
  // No forwarding needed here — doing so would cause the popup to process
  // each update twice.
  if (message.action === 'progressUpdate') {
    sendResponse({});
    return;
  }

  return true;
});

// Handle file downloads
async function handleDownload(url, filename) {
  try {
    await chrome.downloads.download({
      url: url,
      filename: filename,
      saveAs: false
    });
  } catch (error) {
    console.error('Download error:', error);
    
    // Retry with saveAs dialog
    try {
      await chrome.downloads.download({
        url: url,
        filename: filename.split('/').pop(), // Just the filename without folder
        saveAs: true
      });
    } catch (retryError) {
      console.error('Download retry failed:', retryError);
    }
  }
}

// Handle extension installation
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    console.log('WhatsApp Chat Exporter installed');
    
    // Open WhatsApp Web in a new tab on install (optional)
    // chrome.tabs.create({ url: 'https://web.whatsapp.com' });
  } else if (details.reason === 'update') {
    console.log('WhatsApp Chat Exporter updated to version', chrome.runtime.getManifest().version);
  }
});

// Handle extension icon click when not on WhatsApp Web
chrome.action.onClicked.addListener((tab) => {
  if (!tab.url || !tab.url.includes('web.whatsapp.com')) {
    // Open WhatsApp Web if not already there
    chrome.tabs.create({ url: 'https://web.whatsapp.com' });
  }
});

console.log('WhatsApp Chat Exporter background service worker loaded');
