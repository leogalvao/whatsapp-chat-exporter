// WhatsApp Chat Exporter - Popup Script (v1.5 - audit fixes)

document.addEventListener('DOMContentLoaded', async () => {
  const statusDot = document.getElementById('status-dot');
  const statusText = document.getElementById('status-text');
  const chatList = document.getElementById('chat-list');
  const chatCount = document.getElementById('chat-count');
  const selectAllBtn = document.getElementById('select-all-btn');
  const deselectAllBtn = document.getElementById('deselect-all-btn');
  const exportBtn = document.getElementById('export-btn');
  const refreshBtn = document.getElementById('refresh-btn');
  const cancelBtn = document.getElementById('cancel-btn'); // IMP-05
  const progressSection = document.getElementById('progress-section');
  const progressFill = document.getElementById('progress-fill');
  const progressPercent = document.getElementById('progress-percent');
  const progressLabel = document.getElementById('progress-label');
  const progressDetail = document.getElementById('progress-detail');
  const footer = document.querySelector('.footer p');

  // Options
  const includeMedia = document.getElementById('include-media');
  const includeTimestamps = document.getElementById('include-timestamps');
  const includeDeleted = document.getElementById('include-deleted');
  const scrollDepth = document.getElementById('scroll-depth');

  let currentTab = null;
  let isExporting = false;
  let currentBatchIndex = 0;
  let totalBatchSize = 1;
  // BUG-10: Session ID for filtering stale progress updates
  let exportSessionId = null;

  await init();

  // -----------------------------------------------------------------------
  // Initialization
  // -----------------------------------------------------------------------
  async function init() {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      currentTab = tab;

      if (!tab.url || !tab.url.includes('web.whatsapp.com')) {
        setStatus('error', 'Open WhatsApp Web first');
        footer.textContent = 'Navigate to web.whatsapp.com';
        return;
      }

      // IMP-01: Health-check handshake before loading chats
      const health = await healthCheck();
      if (health === 'healthy') {
        await loadChats();
      } else if (health === 'not_injected') {
        // BUG-01: Inject unconditionally, then load chats
        setStatus('loading', 'Injecting content script...');
        try {
          await chrome.scripting.executeScript({
            target: { tabId: currentTab.id },
            files: ['content.js']
          });
          await new Promise(r => setTimeout(r, 500));
          await loadChats();
        } catch (e) {
          setStatus('error', 'Failed to inject content script');
          footer.textContent = e.message;
        }
      } else {
        // listener_dead
        setStatus('error', 'Content script unresponsive');
        footer.textContent = 'Please reload WhatsApp Web and try again';
      }
    } catch (error) {
      console.error('Init error:', error);
      setStatus('error', 'Error connecting');
    }
  }

  // IMP-01: Distinguish between not-injected, listener-dead, and healthy
  async function healthCheck() {
    if (!currentTab || !currentTab.id) return 'not_injected';
    try {
      const response = await withTimeout(
        new Promise((resolve, reject) => {
          chrome.tabs.sendMessage(currentTab.id, { action: 'ping' }, (resp) => {
            if (chrome.runtime.lastError) {
              reject(new Error(chrome.runtime.lastError.message));
            } else {
              resolve(resp);
            }
          });
        }),
        3000
      );
      return (response && response.pong) ? 'healthy' : 'listener_dead';
    } catch (_e) {
      return 'not_injected';
    }
  }

  // -----------------------------------------------------------------------
  // Chat list
  // -----------------------------------------------------------------------
  async function loadChats() {
    setStatus('loading', 'Loading chats...');
    chatList.innerHTML = '<p class="chat-list-placeholder">Loading chats...</p>';
    selectAllBtn.disabled = true;
    deselectAllBtn.disabled = true;
    exportBtn.disabled = true;

    try {
      const response = await sendMessageToContent({ action: 'getChats' });

      if (response && response.success && response.chats && response.chats.length > 0) {
        populateChatList(response.chats);
        setStatus('connected', `${response.chats.length} chats found`);
        footer.textContent = 'Select chats to export';
      } else if (response && response.error === 'DOM_CHANGED') {
        // BUG-02: Show specific DOM error
        setStatus('error', 'WhatsApp layout changed');
        chatList.innerHTML = '<p class="chat-list-placeholder">Extension needs an update for the new WhatsApp Web layout</p>';
        footer.textContent = response.detail || 'DOM selectors no longer match';
      } else {
        // BUG-06: Show the actual error, not just generic text
        const errorDetail = response?.error || response?.detail || '';
        setStatus('error', 'No chats found');
        chatList.innerHTML = `<p class="chat-list-placeholder">No chats available${errorDetail ? ': ' + errorDetail : ''}</p>`;
        footer.textContent = 'Open some chats in WhatsApp';
      }
    } catch (error) {
      console.error('Load chats error:', error);
      // BUG-06: Show actual error string
      setStatus('error', 'Failed to load chats');
      chatList.innerHTML = `<p class="chat-list-placeholder">${error.message || 'Error loading chats'}</p>`;
    }
  }

  function populateChatList(chats) {
    chatList.innerHTML = '';

    chats.forEach((chat, index) => {
      const item = document.createElement('label');
      item.className = 'chat-list-item';
      item.dataset.index = index;

      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.value = index;
      checkbox.dataset.chatName = chat.name || 'Unknown Chat';
      checkbox.dataset.chatId = String(chat.id ?? index);
      checkbox.addEventListener('change', updateSelectionCount);

      const label = document.createElement('span');
      label.className = 'chat-item-name';
      label.textContent = chat.name || 'Unknown Chat';

      // IMP-04: Per-chat progress indicator
      const progressIndicator = document.createElement('span');
      progressIndicator.className = 'chat-item-status';

      item.appendChild(checkbox);
      item.appendChild(label);
      item.appendChild(progressIndicator);
      chatList.appendChild(item);
    });

    selectAllBtn.disabled = false;
    deselectAllBtn.disabled = false;
    updateSelectionCount();
  }

  function getSelectedChats() {
    const checkboxes = chatList.querySelectorAll('input[type="checkbox"]:checked');
    return Array.from(checkboxes).map(cb => ({
      index: parseInt(cb.value),
      name: cb.dataset.chatName,
      id: cb.dataset.chatId
    }));
  }

  function updateSelectionCount() {
    const selected = getSelectedChats();
    chatCount.textContent = `${selected.length} chat${selected.length !== 1 ? 's' : ''} selected`;
    exportBtn.disabled = selected.length === 0 || isExporting;
  }

  // IMP-04: Update per-chat status indicator
  function setChatItemStatus(chatIndex, status) {
    const item = chatList.querySelector(`.chat-list-item[data-index="${chatIndex}"]`);
    if (!item) return;
    const indicator = item.querySelector('.chat-item-status');
    if (!indicator) return;

    item.classList.remove('done', 'exporting', 'failed');
    if (status === 'exporting') {
      item.classList.add('exporting');
      indicator.textContent = '...';
    } else if (status === 'done') {
      item.classList.add('done');
      indicator.textContent = '\u2713'; // checkmark
    } else if (status === 'failed') {
      item.classList.add('failed');
      indicator.textContent = '\u2717'; // X mark
    } else {
      indicator.textContent = '';
    }
  }

  selectAllBtn.addEventListener('click', () => {
    chatList.querySelectorAll('input[type="checkbox"]').forEach(cb => { cb.checked = true; });
    updateSelectionCount();
  });

  deselectAllBtn.addEventListener('click', () => {
    chatList.querySelectorAll('input[type="checkbox"]').forEach(cb => { cb.checked = false; });
    updateSelectionCount();
  });

  function setStatus(status, text) {
    statusDot.className = 'status-dot';
    if (status === 'connected') statusDot.classList.add('connected');
    else if (status === 'error') statusDot.classList.add('error');
    statusText.textContent = text;
  }

  refreshBtn.addEventListener('click', async () => {
    if (!isExporting) await loadChats();
  });

  // IMP-05: Cancel export button
  if (cancelBtn) {
    cancelBtn.addEventListener('click', async () => {
      if (!isExporting) return;
      try {
        await sendMessageToContent({ action: 'cancelExport' });
        updateProgress(0, 'Cancelling...', 'Waiting for current operation to stop');
      } catch (e) {
        console.error('Cancel failed:', e);
      }
    });
  }

  // -----------------------------------------------------------------------
  // Batch export
  // -----------------------------------------------------------------------
  exportBtn.addEventListener('click', async () => {
    const selectedChats = getSelectedChats();
    if (isExporting || selectedChats.length === 0) return;

    isExporting = true;
    totalBatchSize = selectedChats.length;
    // BUG-10: Generate unique session ID for this export batch
    exportSessionId = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    exportBtn.disabled = true;
    refreshBtn.disabled = true;
    chatList.querySelectorAll('input[type="checkbox"]').forEach(cb => { cb.disabled = true; });
    selectAllBtn.disabled = true;
    deselectAllBtn.disabled = true;
    progressSection.classList.remove('hidden');
    if (cancelBtn) cancelBtn.classList.remove('hidden'); // IMP-05

    // Clear previous status markers
    chatList.querySelectorAll('.chat-list-item').forEach(el => setChatItemStatus(el.dataset.index, ''));

    const selectedFormat = document.querySelector('input[name="export-format"]:checked');
    const commonOptions = {
      includeMedia: includeMedia.checked,
      includeTimestamps: includeTimestamps.checked,
      includeDeleted: includeDeleted.checked,
      scrollDepth: scrollDepth.value,
      exportFormat: selectedFormat ? selectedFormat.value : 'json',
      returnData: false // Always download via chrome.downloads
    };

    let successCount = 0;
    let failCount = 0;

    try {
      for (let i = 0; i < selectedChats.length; i++) {
        currentBatchIndex = i;
        const chat = selectedChats[i];
        const tag = `[${i + 1}/${selectedChats.length}]`;

        // IMP-04: Mark chat as currently exporting
        setChatItemStatus(chat.index, 'exporting');

        updateProgress(
          (i / selectedChats.length) * 100,
          `${tag} ${chat.name}`,
          'Starting export...'
        );

        try {
          const options = {
            ...commonOptions,
            chatIndex: chat.index,
            chatName: chat.name
          };

          const response = await sendMessageToContent({
            action: 'exportChat',
            options,
            sessionId: exportSessionId // BUG-10
          });

          if (response && response.success) {
            successCount++;
            setChatItemStatus(chat.index, 'done'); // IMP-04
            // IMP-07: Show export warnings if any
            if (response.warnings && response.warnings.length > 0) {
              console.warn(`Export warnings for ${chat.name}:`, response.warnings);
            }
          } else if (response && response.cancelled) {
            // Export was cancelled — stop the batch
            setChatItemStatus(chat.index, 'failed');
            break;
          } else {
            failCount++;
            setChatItemStatus(chat.index, 'failed'); // IMP-04
            console.error(`Export failed for ${chat.name}:`, response?.error);
          }
        } catch (error) {
          failCount++;
          setChatItemStatus(chat.index, 'failed');
          console.error(`Export error for ${chat.name}:`, error);
        }
      }

      updateProgress(100, 'Batch export complete!',
        `${successCount} exported${failCount > 0 ? `, ${failCount} failed` : ''}`);
      footer.textContent = `Exported ${successCount} of ${selectedChats.length} chats`;

      setTimeout(() => {
        progressSection.classList.add('hidden');
        if (cancelBtn) cancelBtn.classList.add('hidden');
        footer.textContent = 'Select chats to export';
      }, 5000);
    } finally {
      // Always restore UI state even if an unexpected error occurs
      isExporting = false;
      exportSessionId = null;
      exportBtn.disabled = false;
      refreshBtn.disabled = false;
      chatList.querySelectorAll('input[type="checkbox"]').forEach(cb => { cb.disabled = false; });
      selectAllBtn.disabled = false;
      deselectAllBtn.disabled = false;
      updateSelectionCount();
    }
  });

  // -----------------------------------------------------------------------
  // Progress
  // -----------------------------------------------------------------------
  function updateProgress(percent, label, detail) {
    progressFill.style.width = `${percent}%`;
    progressPercent.textContent = `${Math.round(percent)}%`;
    progressLabel.textContent = label;
    progressDetail.textContent = detail;
  }

  // Map per-chat progress updates to overall batch progress
  chrome.runtime.onMessage.addListener((message) => {
    if (message.action === 'progressUpdate' && isExporting) {
      // BUG-10: Ignore progress from a different export session
      if (exportSessionId && message.sessionId && message.sessionId !== exportSessionId) {
        return;
      }
      const overallPercent = ((currentBatchIndex + message.percent / 100) / totalBatchSize) * 100;
      const tag = `[${currentBatchIndex + 1}/${totalBatchSize}]`;
      progressLabel.textContent = `${tag} ${message.label}`;
      progressDetail.textContent = message.detail;
      progressFill.style.width = `${Math.round(overallPercent)}%`;
      progressPercent.textContent = `${Math.round(overallPercent)}%`;
    }
    // Do not return true — we never call sendResponse here, so returning true
    // would leave message channels open indefinitely causing Chrome errors.
  });

  // -----------------------------------------------------------------------
  // Messaging
  // -----------------------------------------------------------------------

  // BUG-06: Timeout wrapper — rejects if content script doesn't respond
  function withTimeout(promise, ms) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`Content script did not respond within ${Math.round(ms / 1000)} s. Reload WhatsApp Web and try again.`));
      }, ms);
      promise.then(
        (val) => { clearTimeout(timer); resolve(val); },
        (err) => { clearTimeout(timer); reject(err); }
      );
    });
  }

  // BUG-01 fix: Always inject content script unconditionally first, then send
  // the message. The content.js guard prevents duplicate listeners during
  // normal navigation, and re-registration is handled on re-injection.
  async function sendMessageToContent(message) {
    if (!currentTab || !currentTab.id) {
      throw new Error('No active tab');
    }

    // Try sending directly first
    try {
      const response = await withTimeout(
        new Promise((resolve, reject) => {
          chrome.tabs.sendMessage(currentTab.id, message, (resp) => {
            if (chrome.runtime.lastError) {
              reject(new Error(chrome.runtime.lastError.message));
            } else {
              resolve(resp);
            }
          });
        }),
        15000 // BUG-06: 15 second timeout
      );
      return response;
    } catch (_firstError) {
      // Content script may not be injected or listener may be dead.
      // Inject unconditionally and retry once.
      try {
        await chrome.scripting.executeScript({
          target: { tabId: currentTab.id },
          files: ['content.js']
        });
      } catch (injectError) {
        throw new Error('Failed to inject content script: ' + injectError.message);
      }

      await new Promise(r => setTimeout(r, 500));

      return withTimeout(
        new Promise((resolve, reject) => {
          chrome.tabs.sendMessage(currentTab.id, message, (resp) => {
            if (chrome.runtime.lastError) {
              reject(new Error(chrome.runtime.lastError.message));
            } else {
              resolve(resp);
            }
          });
        }),
        15000
      );
    }
  }
});
