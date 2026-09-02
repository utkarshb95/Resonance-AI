"""EXPERIMENTAL: standalone Win32 overlay prototype; not used by main.py."""

import ctypes
from ctypes import wintypes
import time

# Define missing types
if not hasattr(wintypes, 'HCURSOR'):
    wintypes.HCURSOR = wintypes.HANDLE

if not hasattr(wintypes, 'LRESULT'):
    wintypes.LRESULT = wintypes.LPARAM

# Use WinDLL instead of windll
user32 = ctypes.WinDLL('user32', use_last_error=True)
gdi32 = ctypes.WinDLL('gdi32', use_last_error=True)
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

# Define proper function signatures
user32.DefWindowProcW.argtypes = [
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM
]
user32.DefWindowProcW.restype = wintypes.LRESULT

# Window procedure callback type
WNDPROCTYPE = ctypes.WINFUNCTYPE(
    wintypes.LRESULT,
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM
)

# Define LRESULT type (not available in wintypes by default)
LRESULT = ctypes.wintypes.LPARAM

# Window procedure callback type
WNDPROCTYPE = ctypes.WINFUNCTYPE(
    LRESULT,
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM
)

# Constants
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_POPUP = 0x80000000
WS_VISIBLE = 0x10000000
LWA_ALPHA = 0x2
WM_PAINT = 0x000F
WM_DESTROY = 0x0002
WM_CLOSE = 0x0010
WM_TIMER = 0x0113
WM_MOUSEMOVE = 0x0200

# Timer IDs
CLOSE_TIMER_ID = 1
FADE_TIMER_ID = 2

# Timeouts
HOVER_TIMEOUT = 10000  # 10 seconds in milliseconds
FADE_DURATION = 3000   # 3 seconds fade duration
FADE_INTERVAL = 100    # Update opacity every 100ms

# Global variables
DEFAULT_ALPHA = 100    # Default opacity (0-255)
current_alpha = DEFAULT_ALPHA
is_fading = False

class WNDCLASS(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROCTYPE),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HCURSOR),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]

class PAINTSTRUCT(ctypes.Structure):
    _fields_ = [
        ("hdc", wintypes.HDC),
        ("fErase", wintypes.BOOL),
        ("rcPaint", wintypes.RECT),
        ("fRestore", wintypes.BOOL),
        ("fIncUpdate", wintypes.BOOL),
        ("rgbReserved", ctypes.c_byte * 32),
    ]

@WNDPROCTYPE
def wnd_proc(hwnd, msg, wparam, lparam):
    global current_alpha, is_fading
    
    if msg == WM_PAINT:
        ps = PAINTSTRUCT()
        hdc = user32.BeginPaint(hwnd, ctypes.byref(ps))
        text = "Overlay Message"
        gdi32.SetTextColor(hdc, 0x00FFFFFF)  # White
        gdi32.SetBkMode(hdc, 1)  # TRANSPARENT
        gdi32.TextOutW(hdc, 20, 20, text, len(text))
        user32.EndPaint(hwnd, ctypes.byref(ps))
        return 0
        
    elif msg == WM_MOUSEMOVE:
        # Mouse is over the window, reset timers and opacity
        print("Mouse move detected!")
        if is_fading:
            # Stop fade effect
            user32.KillTimer(hwnd, FADE_TIMER_ID)
            is_fading = False
            # Reset opacity to default
            current_alpha = DEFAULT_ALPHA
            user32.SetLayeredWindowAttributes(hwnd, 0, current_alpha, LWA_ALPHA)

            # Restore transparent style
            current_style = user32.GetWindowLongW(hwnd, -20)  # GWL_EXSTYLE
            user32.SetWindowLongW(hwnd, -20, current_style | WS_EX_TRANSPARENT)
        
        # Reset the close timer
        user32.KillTimer(hwnd, CLOSE_TIMER_ID)
        user32.SetTimer(hwnd, CLOSE_TIMER_ID, HOVER_TIMEOUT, None)
        return 0
        
    elif msg == WM_TIMER:
        if wparam == CLOSE_TIMER_ID:
            print("Close timer triggered!")
            # Close timer triggered - start fade effect
            user32.KillTimer(hwnd, CLOSE_TIMER_ID)
            is_fading = True

            # Remove transparent style to capture mouse events during fade
            current_style = user32.GetWindowLongW(hwnd, -20)  # GWL_EXSTYLE
            user32.SetWindowLongW(hwnd, -20, current_style & ~WS_EX_TRANSPARENT)

            # Start fade timer
            user32.SetTimer(hwnd, FADE_TIMER_ID, FADE_INTERVAL, None)
            return 0
            
        elif wparam == FADE_TIMER_ID and is_fading:
            # Calculate new alpha for fade effect
            fade_step = DEFAULT_ALPHA / (FADE_DURATION / FADE_INTERVAL)
            current_alpha = max(0, current_alpha - int(fade_step))
            
            # Update window transparency
            user32.SetLayeredWindowAttributes(hwnd, 0, current_alpha, LWA_ALPHA)
            
            # If fully transparent, close the window
            if current_alpha <= 0:
                user32.KillTimer(hwnd, FADE_TIMER_ID)
                user32.PostQuitMessage(0)
            return 0
            
    elif msg == WM_CLOSE:
        user32.DestroyWindow(hwnd)
        return 0
        
    elif msg == WM_DESTROY:
        user32.PostQuitMessage(0)
        return 0
        
    return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

def main():
    hInstance = kernel32.GetModuleHandleW(None)
    className = "OverlayClass"
    wndClass = WNDCLASS()
    wndClass.style = 0
    wndClass.lpfnWndProc = wnd_proc
    wndClass.cbClsExtra = 0
    wndClass.cbWndExtra = 0
    wndClass.hInstance = hInstance
    wndClass.hIcon = None
    wndClass.hCursor = None
    wndClass.hbrBackground = 0
    wndClass.lpszMenuName = None
    wndClass.lpszClassName = className

    if not user32.RegisterClassW(ctypes.byref(wndClass)):
        raise ctypes.WinError()

    hwnd = user32.CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST,
        className,
        "Overlay",
        WS_POPUP | WS_VISIBLE,
        200, 200, 400, 100,
        None, None, hInstance, None
    )
    if not hwnd:
        raise ctypes.WinError()

    # Set initial transparency
    user32.SetLayeredWindowAttributes(hwnd, 0, DEFAULT_ALPHA, LWA_ALPHA)

    # Start the initial close timer
    user32.SetTimer(hwnd, CLOSE_TIMER_ID, HOVER_TIMEOUT, None)

    # Message loop
    msg = wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))

if __name__ == "__main__":
    main()
