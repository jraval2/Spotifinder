from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
#import db, ml, viz
from .db import router as db_router
from .ml import router as ml_router
from .viz import router as viz_router

description = """
Edit your app's title and description. See [https://fastapi.tiangolo.com/tutorial/metadata/](https://fastapi.tiangolo.com/tutorial/metadata/)

To use these interactive docs:
- Click on an endpoint below
- Click the **Try it out** button
- Edit the Request body or any parameters
- Click the **Execute** button
- Scroll down to see the Server response Code & Details
"""

app = FastAPI(
    title='Spotifinder',
    description="Find the next best song for you based on your personal tastes and preferences",
    docs_url='/',
)

app.include_router(db_router, tags=['Database'])
app.include_router(ml_router, tags=['Machine Learning'])
app.include_router(viz_router, tags=['Visualization'])

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

if __name__ == '__main__':
    uvicorn.run(app)
