
import sys
import os
from PySide6 import QtWidgets, QtCore, QtGui

# Add project root to path
sys.path.insert(0, os.getcwd())

def test_page_review_layout():
    app = QtWidgets.QApplication(sys.argv)
    
    print("Testing PageReviewDialog instantiation...")
    try:
        from app.ui.page_review import PageReviewDialog, ResizableLabel
        
        # Mock page record
        page_record = {
            "image_path": "test_image.jpg", 
            "regions": [],
            "output_path": "test_output.jpg"
        }
        
        dialog = PageReviewDialog(page_record=page_record)
        
        # 1. Verify ResizableLabel usage
        if not isinstance(dialog.original_view, ResizableLabel):
            print("FAIL: original_view is not ResizableLabel")
        else:
            print("PASS: original_view is ResizableLabel")
            
        if not isinstance(dialog.translated_view, ResizableLabel):
            print("FAIL: translated_view is not ResizableLabel")
        else:
            print("PASS: translated_view is ResizableLabel")

        # 2. Verify Table Height
        if dialog.table.minimumHeight() != 150:
             print(f"FAIL: Table minimum height is {dialog.table.minimumHeight()}, expected 150")
        else:
             print("PASS: Table minimum height is 150")

        # 3. Check Policy
        policy = dialog.original_view.sizePolicy()
        if policy.horizontalPolicy() == QtWidgets.QSizePolicy.Ignored:
            print("PASS: ResizableLabel policy is Ignored (Correct)")
        else:
            print("FAIL: ResizableLabel policy is NOT Ignored")

    except Exception as e:
        print(f"FAIL: PageReviewDialog error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("PageReviewDialog layout verification completed.")

if __name__ == "__main__":
    test_page_review_layout()
