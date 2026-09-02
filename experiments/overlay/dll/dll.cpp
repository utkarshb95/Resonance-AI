#include <windows.h>

extern "C" __declspec(dllexport) int HideFromCapture(HWND hwnd) {
    if (hwnd != NULL) {
        // Use WDA_EXCLUDEFROMCAPTURE (0x00000002) for Windows 10 v2004+
        DWORD_PTR affinity = 0x00000002;  // WDA_EXCLUDEFROMCAPTURE
        BOOL result = SetWindowDisplayAffinity(hwnd, affinity);
        return result ? 1 : 0;
    }
    return 0;
}

// Add the DLL entry point function
BOOL WINAPI DllMain(
    HINSTANCE hinstDLL,  // handle to DLL module
    DWORD fdwReason,     // reason for calling function
    LPVOID lpReserved )  // reserved
{
    // Perform actions based on the reason for calling
    switch( fdwReason ) 
    { 
        case DLL_PROCESS_ATTACH:
            // Initialize once for each new process
            break;
        case DLL_THREAD_ATTACH:
            // Do thread-specific initialization
            break;
        case DLL_THREAD_DETACH:
            // Do thread-specific cleanup
            break;
        case DLL_PROCESS_DETACH:
            // Perform any necessary cleanup
            break;
    }
    return TRUE;  // Successful DLL_PROCESS_ATTACH
}
