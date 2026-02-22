// WhatsApp Chat Exporter - Popup Script (v1.3 - removed folder picker)

document.addEventListener('DOMContentLoaded', async () => {
  const statusDot = document.getElementById('status-dot');
  const statusText = document.getElementById('status-text');
  const chatList = document.getElementById('chat-list');
  const chatCount = document.getElementById('chat-count');
  const selectAllBtn = document.getElementById('select-all-btn');
  const deselectAllBtn = document.getElementById('deselect-all-btn');
  const exportBtn = document.getElementById('export-btn');
  const refreshBtn = document.getElementById('refresh-btn');
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

      await loadChats();
    } catch (error) {
      console.error('Init error:', error);
      setStatus('error', 'Error connecting');
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
      } else {
        setStatus('error', 'No chats found');
        chatList.innerHTML = '<p class="chat-list-placeholder">No chats available</p>';
        footer.textContent = 'Open some chats in WhatsApp';
      }
    } catch (error) {
      console.error('Load chats error:', error);
      setStatus('error', 'Failed to load chats');
      chatList.innerHTML = '<p class="chat-list-placeholder">Error loading chats</p>';
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
      label.textContent = chat.name || 'Unknown Chat';

      item.appendChild(checkbox);
      item.appendChild(label);
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

  // -----------------------------------------------------------------------
  // Batch export
  // -----------------------------------------------------------------------
  exportBtn.addEventListener('click', async () => {
    const selectedChats = getSelectedChats();
    if (isExporting || selectedChats.length === 0) return;

    isExporting = true;
    totalBatchSize = selectedChats.length;
    exportBtn.disabled = true;
    refreshBtn.disabled = true;
    chatList.querySelectorAll('input[type="checkbox"]').forEach(cb => { cb.disabled = true; });
    selectAllBtn.disabled = true;
    deselectAllBtn.disabled = true;
    progressSection.classList.remove('hidden');

    // Clear previous done markers
    chatList.querySelectorAll('.chat-list-item.done').forEach(el => el.classList.remove('done'));

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
            options
          });

          if (response && response.success) {
            successCount++;
            const item = chatList.querySelector(`.chat-list-item[data-index="${chat.index}"]`);
            if (item) item.classList.add('done');
          } else {
            failCount++;
            console.error(`Export failed for ${chat.name}:`, response?.error);
          }
        } catch (error) {
          failCount++;
          console.error(`Export error for ${chat.name}:`, error);
        }
      }

      updateProgress(100, 'Batch export complete!',
        `${successCount} exported${failCount > 0 ? `, ${failCount} failed` : ''}`);
      footer.textContent = `Exported ${successCount} of ${selectedChats.length} chats`;

      setTimeout(() => {
        progressSection.classList.add('hidden');
        footer.textContent = 'Select chats to export';
      }, 5000);
    } finally {
      // Always restore UI state even if an unexpected error occurs
      isExporting = false;
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
  async function sendMessageToContent(message) {
    return new Promise((resolve, reject) => {
      if (!currentTab || !currentTab.id) {
        reject(new Error('No active tab'));
        return;
      }

      chrome.tabs.sendMessage(currentTab.id, message, (response) => {
        if (chrome.runtime.lastError) {
          // Try injecting content script first
          chrome.scripting.executeScript({
            target: { tabId: currentTab.id },
            files: ['content.js']
          }).then(() => {
            setTimeout(() => {
              chrome.tabs.sendMessage(currentTab.id, message, (retryResponse) => {
                if (chrome.runtime.lastError) {
                  reject(new Error(chrome.runtime.lastError.message));
                } else {
                  resolve(retryResponse);
                }
              });
            }, 500);
          }).catch(reject);
        } else {
          resolve(response);
        }
      });
    });
  }
});