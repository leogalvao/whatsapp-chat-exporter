# WhatsApp Chat Exporter - Chrome Extension

Export your WhatsApp Web chats to JSON with organized media folders.

## Features

- 📱 **Chat Selection** - Dropdown menu to select any chat from your WhatsApp
- 💾 **JSON Export** - Clean, structured JSON format with all message data
- 🖼️ **Media Support** - Downloads images and organizes them in folders
- ⏱️ **Timestamps** - Include or exclude message timestamps
- 🗑️ **Deleted Messages** - Option to include deleted message markers
- 🎨 **Beautiful UI** - Modern, WhatsApp-inspired dark theme

## Installation

### Developer Mode (Unpacked Extension)

1. Download or clone this repository
2. Open Chrome and navigate to `chrome://extensions/`
3. Enable **Developer mode** (toggle in top-right corner)
4. Click **Load unpacked**
5. Select the `whatsapp-exporter` folder
6. The extension icon should appear in your toolbar

## Usage

1. Open [WhatsApp Web](https://web.whatsapp.com) in Chrome
2. Log in and wait for your chats to load
3. Click the extension icon in your toolbar
4. Select a chat from the dropdown menu
5. Configure export options:
   - **Include images & videos** - Download media files
   - **Include timestamps** - Add time to each message
   - **Include deleted messages** - Show "[Message was deleted]" markers
6. Click **Export Chat**
7. Your files will be downloaded automatically

## Export Format

### JSON Structure

```json
{
  "exportInfo": {
    "chatName": "Contact Name",
    "exportDate": "2024-01-15T10:30:00.000Z",
    "totalMessages": 150,
    "totalMedia": 12,
    "exportedBy": "WhatsApp Chat Exporter Extension"
  },
  "messages": [
    {
      "id": "msg_123456789_abc123",
      "type": "text",
      "content": "Hello!",
      "timestamp": "10:30 AM",
      "sender": "You",
      "isOutgoing": true,
      "isDeleted": false,
      "media": null
    },
    {
      "id": "msg_123456790_def456",
      "type": "image",
      "content": "",
      "timestamp": "10:31 AM",
      "sender": "Contact Name",
      "isOutgoing": false,
      "isDeleted": false,
      "media": {
        "type": "image",
        "exportedFilename": "image_1.jpg"
      }
    }
  ],
  "mediaFiles": [
    {
      "messageId": "msg_123456790_def456",
      "type": "image",
      "filename": "image_1.jpg"
    }
  ]
}
```

### File Structure

```
Downloads/
├── whatsapp_Contact_Name_2024-01-15.json
└── whatsapp_Contact_Name_2024-01-15_media/
    ├── image_1.jpg
    ├── image_2.jpg
    └── video_1.mp4
```

## Message Types

- `text` - Regular text messages
- `image` - Photo messages
- `video` - Video messages
- `audio` - Voice messages
- `document` - Document/file attachments
- `deleted` - Deleted message markers

## Troubleshooting

### "No chats found"
- Make sure you're on [web.whatsapp.com](https://web.whatsapp.com)
- Ensure you're logged in and chats are visible
- Click "Refresh Chat List" to reload

### "Failed to load chats"
- Refresh the WhatsApp Web page
- Check if WhatsApp Web is fully loaded
- Try reopening the extension popup

### Media not downloading
- Some media may not be available if not loaded in chat
- Scroll through the chat to load media before exporting
- Check your Chrome download settings

## Privacy & Security

- **Local Only** - All processing happens in your browser
- **No Server** - No data is sent to external servers
- **No Storage** - Extension doesn't store your chat data
- **Open Source** - Full source code available for review

## Limitations

- Only works with WhatsApp Web (not the desktop app)
- Media must be loaded in the chat to be exported
- Very long chats may take time to process
- Some media types may not be fully supported

## Technical Details

- **Manifest Version**: 3
- **Permissions**: 
  - `activeTab` - Access current tab
  - `scripting` - Inject content scripts
  - `downloads` - Save exported files
- **Host Permissions**: `https://web.whatsapp.com/*`

## Version History

### 1.0.0
- Initial release
- Chat selection dropdown
- JSON export with messages
- Media download support
- Export options (timestamps, deleted messages)

## License

MIT License - Feel free to modify and distribute.

## Disclaimer

This extension is not affiliated with WhatsApp or Meta. Use responsibly and respect others' privacy. Only export chats you have permission to export.
