# SuppressAllFillets_Definitive.py
# NXOpen Python Journal
# Finds and non-destructively SUPPRESSES all edge blend (fillet/radius)
# features from the active part.

import NXOpen
import NXOpen.Features

def main():
    the_session = NXOpen.Session.GetSession()
    work_part = the_session.Parts.Work
    lw = the_session.ListingWindow
    lw.Open()
    lw.WriteLine("=== Suppress All Fillets ===")
    
    # A list to hold all the fillet features we find
    fillets_to_suppress = []
    
    # Iterate through all features in the work part
    for feature in work_part.Features:
        # Check if the feature is an Edge Blend (fillet)
        if isinstance(feature, NXOpen.Features.EdgeBlend):
            lw.WriteLine(f"Found fillet feature: {feature.JournalIdentifier}")
            fillets_to_suppress.append(feature)
            
    if not fillets_to_suppress:
        lw.WriteLine("? No fillet features found in the part to suppress.")
        lw.Close()
        return
        
    # --- DEFINITIVE CHANGE: Use SuppressFeatures instead of deletion ---
    try:
        # The SuppressFeatures method is the correct, direct action for this task.
        work_part.Features.SuppressFeatures(fillets_to_suppress)
        lw.WriteLine(f"\nSuccessfully suppressed {len(fillets_to_suppress)} fillet feature(s).")
        
    except NXOpen.NXException as ex:
        lw.WriteLine(f"\nAn NX error occurred during the suppression process: {ex.Message}")
    
    finally:
        lw.WriteLine("\n--- Process Complete ---")
        lw.Close()

if __name__ == '__main__':
    main()
