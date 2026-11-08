"""Full-size, scrollable help and the authoritative keyboard shortcut registry."""

from tkinter import ttk

import tkinter as tk


SHORTCUTS = (
    ("<Control-x>", "Ctrl+X", "Inspect native pixels", "inspect"),
    ("<Control-s>", "Ctrl+S", "Save RAW data / NPZ project", "save"),
    ("<plus>", "+", "Zoom in", "zoom_in"),
    ("<minus>", "−", "Zoom out", "zoom_out"),
    ("<Escape>", "Escape", "Cancel drawing and fit image", "fit"),
    ("<F1>", "F1", "Open help", "help"),
)

EN = """THERMAL STUDIO — USER GUIDE

Navigation
Use the compact, two-row category strip directly above the right settings panel to select View, Filters, Sensor, Palettes, Measurements, RAW editing, Video or Compare. Help and Settings are separate buttons in the header. The strip scrolls horizontally in narrow windows. Drag the vertical divider to resize the settings panel; drag the horizontal divider below the image to resize navigation and readouts. Both control panels have automatic scrollbars. Wheel scrolls a control panel; Shift+wheel scrolls horizontally. All editors and plots stay in this window.

Keyboard shortcuts
{shortcuts}

Mouse and standard controls
Wheel: zoom around the pointer. Left drag: pan when Pan is selected; otherwise draw the active measurement or material mask. Double click: fit the image. On-screen +/−, arrows, Fit and Pixels provide navigation without the wheel. In Compare, keyboard zoom/fit/inspection targets the last image clicked (A by default). Text fields retain normal editing shortcuts. Tab / Shift+Tab moves focus; Space activates the focused button or checkbox; arrow keys navigate a focused list.

Temperature and image sources
Temperature converts native uint16 RAW using °C = RAW/64 − 273.15. Raw counts uses the original codes. Factory brightness is a separate processed 8-bit channel, not a Celsius scale. Filtered temperature smooths only presentation. Six decimal places preserve the encoding, not absolute sensor accuracy. Pixel coordinates stay native through rotation and mirroring.

Colors, scale and focusing
View selects the palette and Auto/Fixed range. Custom palette colors follow Auto/Fixed limits. To retain original Celsius stops, use their endpoints as Fixed limits and disable CLAHE/DDE. White hot / red peak marks the upper temperature range red. CLAHE and DDE change local appearance, so the legend reports observed temperature bands. Focus peaking marks thermal edges in green without modifying measurements or exports.

Measurements and RAW editing
Spot: click. Rectangle and Line: drag between endpoints. Circle: drag from center to radius. Shapes preview while drawing. Select a Line ROI to open its profile. Isotherms highlight an inclusive temperature interval. RAW editing supports emissivity, reflected temperature, air conditions and painted material layers. Enable and apply parameters for frozen/offline correction; live correction additionally requires its explicit experimental switch. This model is not calibrated for P3 and masks do not track motion.

Freeze and file export
Freeze holds the working frame while USB acquisition continues. Open RAW accepts NPZ, native 16-bit PNG and uint16 NPY. Palettes and analysis remain editable after import. Save data preserves RAW and project settings. Save image offers smooth JPEG, RAW PNG and color PNG with optional legend. Return to live leaves offline analysis.

Comparison
Open saved A and B, or use Freeze A/B to copy a displayed comparison slot (the working frame when empty). Live A/B subscribes that slot to incoming camera frames. Freeze A + Live B compares a fixed reference with a changing scene. Loading a file or freezing a live slot replaces only that slot. Both views use a shared range and palette; B − A requires equal sensor dimensions. No image registration is performed. Unplugging clears only live slots and reconnects automatically; frozen and saved references remain intact.

Video
Choose a sampling FPS and Record to a new .p3v file. Stop drains the writer queue; a successful nonempty recording opens automatically. Timeline, frame steps and Play/pause use actual timestamps. Temperature over time plots the selected ROI or whole-frame mean. Experimental FPS cannot create additional camera measurements.

Settings and camera
Settings changes the language immediately and remembers it. English is the default. Sensor can remember mirror per camera model. Use HIGH gain for measurements: LOW has an unresolved temperature offset on tested firmware. X³ is unavailable without a verified driver command. Disconnection does not close the application; closing waits for USB and recording cleanup.
"""

PL = """THERMAL STUDIO — INSTRUKCJA OBSŁUGI

Nawigacja
Kompaktowy, dwurzędowy pasek bezpośrednio nad prawym panelem ustawień przełącza Widok, Filtry, Sensor, Palety, Pomiary, Edycję RAW, Wideo i Porównanie. Pomoc i Ustawienia są osobno w nagłówku. W wąskim oknie pasek przewija się poziomo. Pionowy separator zmienia szerokość ustawień, a poziomy pod obrazem — wysokość nawigacji i odczytów. Oba panele mają automatyczne paski przewijania. Kółko przewija panel kontrolek; Shift+kółko przewija poziomo. Edytory i wykresy pozostają w jednym oknie.

Skróty klawiszowe
{shortcuts}

Mysz i standardowe kontrolki
Kółko: powiększanie wokół kursora. Przeciąganie lewym przyciskiem: przesuwanie w narzędziu Przesuwanie, a w innych narzędziach rysowanie pomiaru lub maski materiału. Dwuklik: dopasowanie obrazu. Przyciski +/−, strzałki, Dopasuj i Piksele pozwalają nawigować bez kółka. W Porównaniu skróty powiększenia, dopasowania i inspekcji dotyczą ostatnio klikniętego obrazu (domyślnie A). Pola tekstowe zachowują standardowe skróty edycji. Tab / Shift+Tab zmienia fokus; Spacja aktywuje przycisk lub przełącznik; strzałki obsługują aktywną listę.

Temperatura i źródła obrazu
Temperatura przelicza natywne uint16 RAW według °C = RAW/64 − 273.15. Wartości RAW pokazują oryginalne kody. Jasność fabryczna jest osobnym przetworzonym kanałem 8-bitowym, nie skalą °C. Temperatura filtrowana wygładza tylko prezentację. Sześć miejsc po przecinku zachowuje kodowanie, a nie dokładność sensora. Współrzędne pozostają natywne po obrocie i odbiciu.

Kolory, skala i ostrość
Widok pozwala wybrać paletę oraz zakres automatyczny lub stały. Palety użytkownika podlegają zakresowi Auto/Fixed. Aby zachować progi w °C, ustaw granice Fixed równe końcom palety i wyłącz CLAHE/DDE. Biel / czerwony szczyt oznacza górny zakres temperatur czerwienią. CLAHE i DDE zmieniają lokalny wygląd, więc legenda opisuje zaobserwowane pasma temperatur. Podświetlanie krawędzi zaznacza je na zielono, bez zmiany pomiarów i eksportów.

Pomiary i edycja RAW
Punkt: kliknij. Prostokąt i Linia: przeciągnij między końcami. Koło: przeciągnij od środka do promienia. Figury mają podgląd podczas rysowania. Zaznacz linię w tabeli, aby otworzyć profil. Izotermy zaznaczają domknięty przedział temperatur. Edycja RAW obsługuje emisyjność, temperaturę odbitą, warunki powietrza i malowane warstwy materiałów. Włącz i zastosuj parametry korekcji; strumień na żywo wymaga dodatkowego jawnego przełącznika eksperymentalnego. Model nie jest skalibrowany dla P3, a maski nie śledzą ruchu.

Zamrażanie i eksport
Zamroź zatrzymuje klatkę roboczą, podczas gdy USB nadal ją odczytuje. Otwórz RAW przyjmuje NPZ, natywny 16-bitowy PNG i uint16 NPY. Po imporcie nadal można zmieniać palety i analizę. Zapis danych zachowuje RAW i ustawienia projektu. Zapis obrazu oferuje wygładzony JPEG, PNG RAW i kolorowy PNG z opcjonalną legendą. Powrót na żywo kończy analizę offline.

Porównanie
Otwórz zapisane A i B albo użyj Zamroź A/B, aby skopiować wyświetlaną klatkę danego pola (klatkę roboczą, jeśli pole jest puste). Na żywo A/B podłącza pole do strumienia kamery. Zamroź A + Na żywo B porównuje stały wzorzec ze zmieniającą się sceną. Import pliku lub zamrożenie zmienia tylko wybrane pole. Oba obrazy mają wspólną skalę i paletę; B − A wymaga jednakowych wymiarów sensora. Nie ma automatycznego dopasowania scen. Odłączenie usuwa tylko podgląd na żywo; odnawianie połączenia jest automatyczne, a zapisane i zamrożone wzorce pozostają.

Wideo
Wybierz częstotliwość próbkowania i nagraj nowy plik .p3v. Zatrzymanie opróżnia kolejkę zapisu; udane niepuste nagranie otwiera się automatycznie. Suwak, kroki klatek i odtwarzanie korzystają z rzeczywistych czasów. Wykres czasowy pokazuje średnią wybranego ROI lub całej klatki. Eksperymentalne FPS nie tworzy dodatkowych pomiarów kamery.

Ustawienia i kamera
Ustawienia zmieniają język od razu i zapamiętują wybór; domyślny jest angielski. Sensor może zapamiętać odbicie dla modelu kamery. Do pomiarów używaj HIGH: LOW ma nierozwiązane przesunięcie temperatur na testowanym firmware. X³ pozostaje niedostępny bez potwierdzonego polecenia sterownika. Odłączenie nie zamyka aplikacji; zamknięcie czeka na zakończenie USB i zapisu.
"""


class HelpPanel(ttk.Frame):
    def __init__(self, parent, translator, on_close):
        super().__init__(parent, padding=18)
        self.translator = translator
        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 10))
        ttk.Button(header, text="Close help", command=on_close).pack(side="right")
        scrollbar = ttk.Scrollbar(self)
        scrollbar.pack(side="right", fill="y")
        self.text = tk.Text(
            self,
            wrap="word",
            background="#18212e",
            foreground="#e7edf5",
            font=("TkDefaultFont", 12),
            padx=20,
            pady=16,
            yscrollcommand=scrollbar.set,
        )
        self.text.pack(fill="both", expand=True)
        scrollbar.configure(command=self.text.yview)
        self.refresh()

    def refresh(self):
        shortcuts = "\n".join(
            f"{key} — {self.translator.text(label)}" for _, key, label, _ in SHORTCUTS
        )
        content = (PL if self.translator.language == "pl" else EN).format(
            shortcuts=shortcuts
        )
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", content)
        self.text.configure(state="disabled")
