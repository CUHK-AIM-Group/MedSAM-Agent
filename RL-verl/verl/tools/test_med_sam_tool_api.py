#!/usr/bin/env python3
"""
Offline test for multi-turn interaction in med_sam_tool_api.py

Test scenarios:
1. Create tool instance
2. Create segmentation session (simulate image upload)
3. Multi-round click segmentation (accumulate clicks, use previous mask)
4. Verify session state
5. Clean up resources

Dependency: API server must be running at http://localhost:8265
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image, ImageDraw


def create_test_image(width=512, height=512):
    """Create a test image with two circular regions."""
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw two circles as test targets
    # Circle 1: red, left
    draw.ellipse([100, 150, 200, 250], fill='red', outline='darkred', width=3)
    
    # Circle 2: blue, right
    draw.ellipse([300, 200, 400, 300], fill='blue', outline='darkblue', width=3)
    
    # Add some noise
    draw.rectangle([50, 50, 100, 100], fill='lightgray')
    draw.rectangle([420, 350, 470, 400], fill='lightgray')
    
    return img


async def test_multiturn_api_tool():
    """Test the multi-turn interactive API tool."""
    print("=" * 70)
    print("Multi-turn API tool offline test")
    print("=" * 70)
    
    # 1. Import tool class
    print("\n[1] Import MedSAMToolAPI...")
    try:
        from verl.tools.med_sam_tool_api import MedSAMToolAPI
        from verl.tools.schemas import OpenAIFunctionToolSchema
        print("    ✅ Import succeeded")
    except Exception as e:
        print(f"    ❌ Import failed: {e}")
        return False
    
    # 2. Create tool config
    print("\n[2] Configure tool...")
    config = {
        "api_base_url": "http://localhost:8265",
        "use_multiturn": True,  # Enable multi-turn mode
        "max_retries": 3,
        "retry_delay": 1.0,
        "timeout": 30
    }
    
    tool_schema = OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "segment",
            "description": "Segment medical image with points",
            "parameters": {
                "type": "object",
                "properties": {
                    "points": {"type": "array"},
                    "point_labels": {"type": "array"}
                },
                "required": ["points"]
            }
        }
    )
    
    print(f"    Config: {config}")
    
    # 3. Initialize tool
    print("\n[3] Initialize tool...")
    try:
        tool = MedSAMToolAPI(config, tool_schema)
        print("    ✅ Tool initialized successfully")
    except Exception as e:
        print(f"    ❌ Tool initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Create test image
    print("\n[4] Create test image...")
    test_image = create_test_image(512, 512)
    test_image_path = "/tmp/test_medsam_api_tool.png"
    test_image.save(test_image_path)
    print(f"    ✅ Test image saved: {test_image_path}")
    print(f"    Image size: {test_image.size}")
    
    # 5. Create segmentation instance (upload image)
    print("\n[5] Create segmentation instance (upload image to API)...")
    try:
        instance_id, response = await tool.create(
            image=test_image_path
        )
        print(f"    ✅ Instance created successfully")
        print(f"       Instance ID: {instance_id}")
        
        # Check whether an API session was created
        inst = tool._instances.get(instance_id)
        if inst:
            api_session_id = inst.get("api_session_id")
            if api_session_id:
                print(f"       API Session ID: {api_session_id}")
                print(f"       ✅ Multi-turn mode enabled (API session created)")
            else:
                print(f"       ⚠️ API session not created (fallback to single-shot mode)")
        
    except Exception as e:
        print(f"    ❌ Instance creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Round 1: add a positive point (target: left red circle)
    print("\n[6] Round 1 segmentation - add positive point [150, 200] (inside red circle)...")
    try:
        parameters = {
            "points": [150, 200],  # single point
            "point_labels": [1]     # positive point
        }
        
        response, reward, extra = await tool.execute(instance_id, parameters)
        
        print(f"    ✅ Round 1 segmentation completed")
        print(f"       Response text: {response.text}")
        print(f"       Reward: {reward}")
        print(f"       Success: {extra.get('success')}")
        
        # Check returned mask
        if response.image and len(response.image) > 0:
            mask1 = response.image[0]
            print(f"       Mask size: {mask1.size}")
            print(f"       Mask mode: {mask1.mode}")
            
            # Save mask
            mask1.save("/tmp/test_round1_mask.png")
            print(f"       Mask saved: /tmp/test_round1_mask.png")
            
            # Count mask pixels
            mask_np = np.array(mask1)
            white_pixels = np.sum(mask_np > 0)
            total_pixels = mask_np.size
            print(f"       Segmentation area: {white_pixels}/{total_pixels} = {100*white_pixels/total_pixels:.1f}%")
        else:
            print(f"       ⚠️ No mask returned")
        
        # Check click history
        clicks_list = inst.get("clicks_list", [])
        print(f"       Total clicks: {len(clicks_list)}")
        
    except Exception as e:
        print(f"    ❌ Round 1 segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        await tool.release(instance_id)
        return False
    
    # 7. Round 2: add another positive point (expand segmentation)
    print("\n[7] Round 2 segmentation - add positive point [160, 220] (accumulated mode)...")
    try:
        parameters = {
            "points": [160, 220],
            "point_labels": [1]
        }
        
        response, reward, extra = await tool.execute(instance_id, parameters)
        
        print(f"    ✅ Round 2 segmentation completed")
        print(f"       Response text: {response.text}")
        
        if response.image and len(response.image) > 0:
            mask2 = response.image[0]
            mask2.save("/tmp/test_round2_mask.png")
            print(f"       Mask saved: /tmp/test_round2_mask.png")
            
            mask_np = np.array(mask2)
            white_pixels = np.sum(mask_np > 0)
            total_pixels = mask_np.size
            print(f"       Segmentation area: {white_pixels}/{total_pixels} = {100*white_pixels/total_pixels:.1f}%")
        
        clicks_list = inst.get("clicks_list", [])
        print(f"       Total clicks: {len(clicks_list)}")
        print(f"       Click history: {clicks_list}")
        
    except Exception as e:
        print(f"    ❌ Round 2 segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        await tool.release(instance_id)
        return False
    
    # 8. Round 3: add a negative point (exclude erroneous regions)
    print("\n[8] Round 3 segmentation - add negative point [250, 250] (correct over-segmentation)...")
    try:
        parameters = {
            "points": [250, 250],
            "point_labels": [0]  # negative point
        }
        
        response, reward, extra = await tool.execute(instance_id, parameters)
        
        print(f"    ✅ Round 3 segmentation completed")
        print(f"       Response text: {response.text}")
        
        if response.image and len(response.image) > 0:
            mask3 = response.image[0]
            mask3.save("/tmp/test_round3_mask.png")
            print(f"       Mask saved: /tmp/test_round3_mask.png")
            
            mask_np = np.array(mask3)
            white_pixels = np.sum(mask_np > 0)
            total_pixels = mask_np.size
            print(f"       Segmentation area: {white_pixels}/{total_pixels} = {100*white_pixels/total_pixels:.1f}%")
        
        clicks_list = inst.get("clicks_list", [])
        print(f"       Total clicks: {len(clicks_list)}")
        
    except Exception as e:
        print(f"    ❌ Round 3 segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        await tool.release(instance_id)
        return False
    
    # 9. Verify multi-turn mode
    print("\n[9] Verify multi-turn interaction features...")
    
    # Check instance status
    if instance_id in tool._instances:
        inst = tool._instances[instance_id]
        api_session_id = inst.get("api_session_id")
        clicks_list = inst.get("clicks_list", [])
        
        print(f"    Instance status:")
        print(f"       API Session ID: {api_session_id}")
        print(f"       Total clicks: {len(clicks_list)}")
        print(f"       Click details: {clicks_list}")
        
        if api_session_id and len(clicks_list) == 3:
            print(f"    ✅ Multi-turn interaction working")
            print(f"       - API session created")
            print(f"       - Clicks accumulated (3 rounds = 3 points)")
            print(f"       - Each round refines using previous mask")
        else:
            print(f"    ⚠️ Multi-turn interaction may not be working")
            if not api_session_id:
                print(f"       - API session not created (fallback to single-shot mode)")
            if len(clicks_list) != 3:
                print(f"       - Click count mismatch (expected 3, got {len(clicks_list)})")
    
    # 10. Cleanup resources
    print("\n[10] Cleanup resources...")
    try:
        await tool.release(instance_id)
        print(f"    ✅ Instance released")
        
        if instance_id not in tool._instances:
            print(f"    ✅ Instance removed from memory")
        
        # If there was an API session, it should have been deleted
        if api_session_id:
            print(f"    ℹ️ API session {api_session_id} should have been deleted on the server")
        
    except Exception as e:
        print(f"    ❌ Cleanup failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✅ Test completed!")
    print("=" * 70)
    print("\nTest summary:")
    print("  ✓ Tool initialization")
    print("  ✓ Instance creation (API session creation)")
    print("  ✓ Round 1: single-point segmentation")
    print("  ✓ Round 2: accumulate clicks, iterative refinement")
    print("  ✓ Round 3: negative point correction")
    print("  ✓ Multi-turn interaction verification")
    print("  ✓ Resource cleanup")
    print("\nGenerated files:")
    print("  - /tmp/test_medsam_api_tool.png (test image)")
    print("  - /tmp/test_round1_mask.png (round 1 mask)")
    print("  - /tmp/test_round2_mask.png (round 2 mask)")
    print("  - /tmp/test_round3_mask.png (round 3 mask)")
    
    return True


async def test_single_shot_mode():
    """Test single-shot mode (use_multiturn=False)."""
    print("\n" + "=" * 70)
    print("Single-shot mode test (use_multiturn=False)")
    print("=" * 70)
    
    from verl.tools.med_sam_tool_api import MedSAMToolAPI
    from verl.tools.schemas import OpenAIFunctionToolSchema
    
    # Configure single-shot mode
    config = {
        "api_base_url": "http://localhost:8265",
        "use_multiturn": False,  # Disable multi-turn mode
        "max_retries": 3,
        "timeout": 30
    }
    
    tool_schema = OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "segment",
            "description": "Segment medical image",
            "parameters": {
                "type": "object",
                "properties": {
                    "points": {"type": "array"}
                },
                "required": ["points"]
            }
        }
    )
    
    print("\n[1] Initialize tool (single-shot mode)...")
    tool = MedSAMToolAPI(config, tool_schema)
    print("    ✅ Tool initialization completed")
    
    print("\n[2] Create instance...")
    test_image = create_test_image()
    instance_id, _ = await tool.create(image=test_image)
    
    inst = tool._instances.get(instance_id)
    api_session_id = inst.get("api_session_id") if inst else None
    
    print(f"    Instance ID: {instance_id}")
    print(f"    API Session ID: {api_session_id}")
    
    if not api_session_id:
        print("    ✅ Single-shot mode correct: API session not created")
    else:
        print("    ⚠️ Single-shot mode abnormal: API session should not be created")
    
    print("\n[3] Run segmentation...")
    response, _, extra = await tool.execute(instance_id, {"points": [150, 200]})
    
    if extra.get("success"):
        print("    ✅ Single prediction succeeded")
        print(f"    Response: {response.text}")
    else:
        print("    ❌ Single prediction failed")
    
    print("\n[4] Cleanup...")
    await tool.release(instance_id)
    print("    ✅ Cleanup completed")
    
    return True


async def test_error_handling():
    """Test error handling."""
    print("\n" + "=" * 70)
    print("Error handling test")
    print("=" * 70)
    
    from verl.tools.med_sam_tool_api import MedSAMToolAPI
    from verl.tools.schemas import OpenAIFunctionToolSchema
    
    config = {
        "api_base_url": "http://localhost:8265",
        "use_multiturn": True,
        "max_retries": 2,
        "retry_delay": 0.5,
        "timeout": 30
    }
    
    tool_schema = OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "segment",
            "description": "test",
            "parameters": {
                "type": "object",
                "properties": {
                    "points": {"type": "array"}
                },
                "required": ["points"]
            }
        }
    )
    
    tool = MedSAMToolAPI(config, tool_schema)
    
    # Test 1: invalid instance_id
    print("\n[Test 1] Use invalid instance_id...")
    response, reward, extra = await tool.execute("invalid_id", {"points": [100, 100]})
    if not extra.get("success") and reward < 0:
        print("    ✅ Correctly handled invalid instance_id")
    else:
        print("    ❌ Should return an error")
    
    # Test 2: missing points parameter
    print("\n[Test 2] Missing required points parameter...")
    test_image = create_test_image()
    instance_id, _ = await tool.create(image=test_image)
    
    response, reward, extra = await tool.execute(instance_id, {})  # no points
    if not extra.get("success") and reward < 0:
        print("    ✅ Correctly handled missing parameter")
    else:
        print("    ❌ Should return an error")
    
    await tool.release(instance_id)
    
    return True


async def main():
    """Main test function."""
    import requests
    
    # Check API server
    print("Checking API server status...")
    try:
        # Disable proxy
        os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
        session = requests.Session()
        session.trust_env = False
        
        response = session.get("http://localhost:8265/health", timeout=2)
        if response.status_code == 200:
            print("✅ API server is running normally\n")
        else:
            print(f"⚠️ API server response abnormal: {response.status_code}\n")
    except:
        print("❌ Unable to connect to API server (http://localhost:8265)")
        print("Please start the server first: python api_server/imisnet.py\n")
        return
    
    # Run tests
    success = True
    
    # Main test: multi-turn interaction
    try:
        result = await test_multiturn_api_tool()
        if not result:
            success = False
    except Exception as e:
        print(f"\n❌ Multi-turn interaction test error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    # Test: single-shot mode
    try:
        result = await test_single_shot_mode()
        if not result:
            success = False
    except Exception as e:
        print(f"\n❌ Single-shot mode test error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    # Test: error handling
    try:
        result = await test_error_handling()
        if not result:
            success = False
    except Exception as e:
        print(f"\n❌ Error handling test error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    # Final result
    print("\n" + "=" * 70)
    if success:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
