# PSIO_Projekt
CyberTrener/
├── data/                   # Dane do treningu i testów
│   ├── raw_video/          # Nagrania wzorcowe i błędne (z IP Webcam) [cite: 20]
│   ├── dataset_yolo/       # Zdjęcia sztangi opisane w Roboflow 
│   └── database/           # Plik bazy danych treningi.db 
├── models/                 # Przechowywanie modeli AI
│   └── overhead_press.pt   # Wyeksportowane wagi z YOLO v11 
├── src/                    # Kod źródłowy aplikacji
│   ├── __init__.py
│   ├── main.py             # Główny plik uruchamiający system [cite: 29]
│   ├── camera_handler.py   # Obsługa IP Webcam i kamery laptopa [cite: 27]
│   ├── pose_analysis.py    # Autorskie funkcje matematyczne i MediaPipe 
│   ├── exercise_logic.py   # Logika OHP (liczenie powtórzeń, błędy) 
│   ├── database_manager.py # Obsługa bazy SQL (zapisywanie sesji) 
│   └── ui_display.py       # Interfejs graficzny w Pygame [cite: 23, 31]
├── reports/                # Dokumentacja i raporty
│   ├── research_article.pdf # Końcowy raport w formie artykułu 
│   └── tests/              # Wyniki testów dokładności i wydajności [cite: 33]
├── requirements.txt        # Lista bibliotek (opencv, mediapipe, ultralytics itd.)
└── README.md               # Instrukcja uruchomienia i opis projektu
