#!/usr/bin/env python3
"""
Debug script to test loss function behavior.
"""

import torch
import torch.nn as nn

def test_loss_function():
    """Test the loss function behavior."""
    print("🔍 Testing Loss Function Behavior")
    print("="*50)
    
    # Create sample predictions (all tensors)
    batch_size = 2
    num_classes = 701
    
    predictions = [
        torch.randn(batch_size, num_classes),  # global
        torch.randn(batch_size, num_classes),  # regional 1
        torch.randn(batch_size, num_classes),  # regional 2
        torch.randn(batch_size, num_classes),  # regional 3
        torch.randn(batch_size, num_classes),  # final
    ]
    
    labels = torch.randint(0, num_classes, (batch_size,))
    
    print(f"📊 Test Data:")
    print(f"   Batch size: {batch_size}")
    print(f"   Num classes: {num_classes}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Predictions: {len(predictions)} tensors")
    
    for i, pred in enumerate(predictions):
        print(f"     Pred {i}: type={type(pred)}, shape={pred.shape}")
    
    # Test classification loss
    print(f"\n🧪 Testing Classification Loss:")
    criterion = nn.CrossEntropyLoss()
    
    try:
        total_loss = 0.0
        for i, pred in enumerate(predictions):
            # This is what the loss function does
            if isinstance(pred, list):
                print(f"   Pred {i} is a list (unexpected!)")
                pred_tensor = pred[0]
            else:
                print(f"   Pred {i} is a tensor (expected)")
                pred_tensor = pred
            
            loss = criterion(pred_tensor, labels)
            total_loss += loss
            print(f"     Loss {i}: {loss.item():.4f}")
        
        print(f"   Total loss: {total_loss.item():.4f}")
        print(f"✅ Loss computation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_problematic_case():
    """Test the problematic case that might cause the error."""
    print("\n🔍 Testing Problematic Case")
    print("="*40)
    
    # Simulate what might be happening
    batch_size = 8  # Same as training
    num_classes = 701
    
    # Create predictions that might have mixed formats
    predictions = [
        torch.randn(batch_size, num_classes),  # Normal tensor
        [torch.randn(batch_size, num_classes), torch.randn(batch_size, 256)],  # List format
        torch.randn(batch_size, num_classes),  # Normal tensor
    ]
    
    labels = torch.randint(0, num_classes, (batch_size,))
    
    print(f"📊 Problematic Test Data:")
    print(f"   Batch size: {batch_size}")
    print(f"   Labels shape: {labels.shape}")
    
    for i, pred in enumerate(predictions):
        print(f"   Pred {i}: type={type(pred)}")
        if isinstance(pred, list):
            print(f"     List length: {len(pred)}")
            for j, item in enumerate(pred):
                print(f"       Item {j}: type={type(item)}, shape={item.shape}")
        else:
            print(f"     Shape: {pred.shape}")
    
    # Test loss computation
    print(f"\n🧪 Testing Loss with Mixed Formats:")
    criterion = nn.CrossEntropyLoss()
    
    try:
        total_loss = 0.0
        for i, pred in enumerate(predictions):
            print(f"   Processing pred {i}...")
            
            if isinstance(pred, list):
                print(f"     Pred is list, taking first element")
                pred_tensor = pred[0]
            else:
                print(f"     Pred is tensor")
                pred_tensor = pred
            
            print(f"     Pred tensor shape: {pred_tensor.shape}")
            print(f"     Labels shape: {labels.shape}")
            
            loss = criterion(pred_tensor, labels)
            total_loss += loss
            print(f"     Loss {i}: {loss.item():.4f}")
        
        print(f"   Total loss: {total_loss.item():.4f}")
        print(f"✅ Mixed format loss computation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Mixed format loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tensor_concatenation():
    """Test tensor concatenation that might cause the error."""
    print("\n🔍 Testing Tensor Concatenation")
    print("="*40)
    
    try:
        # This might be what's causing the error
        batch_size = 8
        
        # Different sized tensors
        tensor1 = torch.randn(batch_size, 256)  # Size 8
        tensor2 = torch.randn(batch_size * 2, 256)  # Size 16 (problematic!)
        
        print(f"   Tensor 1 shape: {tensor1.shape}")
        print(f"   Tensor 2 shape: {tensor2.shape}")
        
        # This would cause the error we're seeing
        try:
            concatenated = torch.cat([tensor1, tensor2], dim=1)
            print(f"   Concatenated shape: {concatenated.shape}")
        except Exception as e:
            print(f"   ❌ Concatenation failed: {e}")
            print(f"   This matches our error message!")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Tensor concatenation test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🎯 Loss Function Debug Session")
    print("="*50)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Normal case
    if test_loss_function():
        success_count += 1
    
    # Test 2: Problematic case
    if test_problematic_case():
        success_count += 1
    
    # Test 3: Tensor concatenation
    if test_tensor_concatenation():
        success_count += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"🎊 DEBUG SUMMARY")
    print(f"{'='*50}")
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count >= 2:
        print(f"🎉 Found potential issue!")
        print(f"💡 The error might be caused by:")
        print(f"   - Batch size mismatch in tensor concatenation")
        print(f"   - Mixed tensor formats in predictions")
        print(f"   - Dimension mismatch in feature fusion")
    else:
        print(f"⚠️  Need more investigation.")
    
    return success_count >= 2


if __name__ == "__main__":
    main()
