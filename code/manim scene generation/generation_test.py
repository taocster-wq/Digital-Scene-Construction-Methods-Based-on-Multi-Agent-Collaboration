# 生成测试
import asyncio
from generation import generation_video

async def main():
    topic = "draw a triangle"
    description=f"""
Draw a triangle with three sides and three angles. The triangle should be equilateral, meaning all sides are of equal length and all angles are equal to 60 degrees.
    """
    difficulty="Easy"
    result = await generation_video(topic=topic, description=description,difficulty=difficulty,max_retries=5)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
