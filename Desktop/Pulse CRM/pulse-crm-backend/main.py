from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, companies, contacts, leads, deals, timeline, gmail, users

app = FastAPI(
    title="Pulse CRM REST API",
    description="Backend business logic and synchronization controllers for Pulse CRM",
    version="1.0.0"
)

# CORS configurations for cross-origin frontend browser calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adapt to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all endpoint routers
app.include_router(auth.router)
app.include_router(companies.router)
app.include_router(contacts.router)
app.include_router(leads.router)
app.include_router(deals.router)
app.include_router(timeline.router)
app.include_router(gmail.router)
app.include_router(users.router)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Pulse CRM REST API",
        "version": "1.0.0",
        "documentation": "/docs"
    }
