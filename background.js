// WhatsApp Chat Exporter - Background Service Worker
// Handles downloads and other background tasks

// Listen for messages from content script and popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // BUG-03 fix: Only return true for async handlers; close channel immediately
  // for everything else to avoid "async response never sent" warnings.
  if (message.action === 'download') {
    handleDownload(message.url, message.filename)
      .then(() => sendResponse({ success: true }))
      .catch(e => sendResponse({ success: false, error: e.message }));
    return true; // async — keep channel open
  }

  // progressUpdate messages are sent by the content script directly to all
  // extension views (including the popup) via chrome.runtime.sendMessage.
  // No forwarding needed here.
  if (message.action === 'progressUpdate') {
    // No response needed — return false (default) to close channel
    return;
  }

  // Unknown action — close channel immediately
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
      throw retryError;
    }
  }
}

// Handle extension installation
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    console.log('WhatsApp Chat Exporter installed');
  } else if (details.reason === 'update') {
    console.log('WhatsApp Chat Exporter updated to version', chrome.runtime.getManifest().version);
  }
});

// BUG-08 fix: Removed dead action.onClicked handler — the manifest declares
// default_popup so onClicked never fires. If a "navigate to WhatsApp" action
// is desired, it should be a button in the popup UI instead.

console.log('WhatsApp Chat Exporter background service worker loaded');
