import asyncio
import websockets

async def test_ws():
    uri = "wss://system-mediapipe-1.onrender.com/ws/pushups"
    try:
        print(f"Connecting to {uri}...")
        async with websockets.connect(uri) as websocket:
            print("Connected successfully!")
            await websocket.send("Hello")
            print("Message sent.")
    except Exception as e:
        print(f"Failed: {e}")

asyncio.run(test_ws())
