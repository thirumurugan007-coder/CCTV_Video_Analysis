# CCTV Video Analysis System

## Description
AI-powered video detection and analysis system for CCTV surveillance using YOLOv8/v9.

## Features
- Real-time object detection
- Object tracking
- Video analysis
- RESTful API
- WebSocket streaming
- Custom model training

## Installation Instructions
To install the project, clone the repository and install the required packages:
```bash
git clone https://github.com/thirumurugan007-coder/CCTV_Video_Analysis.git
cd CCTV_Video_Analysis
pip install -r requirements.txt
```

## Quick Start Guide
To start the application, run:
```bash
python main.py
```

## API Usage Examples
### Uploading Videos
```bash
curl -X POST http://localhost:8000/upload -F "file=@your_video.mp4"
```
### Detecting Objects
```bash
curl -X POST http://localhost:8000/detect -F "file=@your_video.mp4"
```

## Training Custom Models
To train custom models, use the following command:
```bash
python train.py --data data.yaml --cfg yolov8.yaml --weights yolov8.pt
```

## Configuration
Create a `.env` file with the following example configuration:
```
API_KEY=your_api_key
DEBUG=True
```

## Project Structure Diagram
```
CCTV_Video_Analysis/
├── main.py
├── train.py
├── requirements.txt
├── .env
└── ...
```

## API Endpoints
| Endpoint                | Method | Description              |
|------------------------|--------|--------------------------|
| /upload                | POST   | Upload a video           |
| /detect                | POST   | Detect objects in video  |

## Technologies Used
- FastAPI
- Ultralytics YOLOv8
- OpenCV
- PyTorch

## License
MIT License

## Author
Thirumurugan K

## Contributing Guidelines
Please open an issue to discuss any changes you'd like to make.

## Support
For support, please open an issue on GitHub or contact the author directly.