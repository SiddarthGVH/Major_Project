from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, companies, contacts, leads, deals, timeline, gmail, users

from database import engine
import models

app = FastAPI(
    title="Pulse CRM REST API",
    description="Backend business logic and synchronization controllers for Pulse CRM",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    # Create all tables if they don't exist
    models.Base.metadata.create_all(bind=engine)
    
    # Seed the database
    from database import SessionLocal
    from uuid import UUID
    
    db = SessionLocal()
    try:
        # Seed Roles
        if db.query(models.Role).first() is None:
            roles = [
                models.Role(
                    id=UUID('a8f90c42-b0c6-4767-88ea-d4b68e9f2915'),
                    name='Administrator',
                    description='Full system access and configurations control'
                ),
                models.Role(
                    id=UUID('b1f80c42-b0c6-4767-88ea-d4b68e9f2916'),
                    name='Sales Manager',
                    description='Auditing, tracking pipelines, and signing off on deals'
                ),
                models.Role(
                    id=UUID('c2f70c42-b0c6-4767-88ea-d4b68e9f2917'),
                    name='Sales Representative',
                    description='Frontline workspace access to log and sync client pipelines'
                )
            ]
            db.add_all(roles)
            db.commit()

        # Seed Pipeline Stages
        if db.query(models.PipelineStage).first() is None:
            stages = [
                models.PipelineStage(
                    id=UUID('d1f60c42-b0c6-4767-88ea-d4b68e9f2918'),
                    name='Qualified',
                    probability=10.00,
                    display_order=1
                ),
                models.PipelineStage(
                    id=UUID('e2f50c42-b0c6-4767-88ea-d4b68e9f2919'),
                    name='Proposal',
                    probability=40.00,
                    display_order=2
                ),
                models.PipelineStage(
                    id=UUID('f3f40c42-b0c6-4767-88ea-d4b68e9f2920'),
                    name='Under Review',
                    probability=70.00,
                    display_order=3
                ),
                models.PipelineStage(
                    id=UUID('a4f30c42-b0c6-4767-88ea-d4b68e9f2921'),
                    name='Won',
                    probability=100.00,
                    display_order=4
                ),
                models.PipelineStage(
                    id=UUID('b5f20c42-b0c6-4767-88ea-d4b68e9f2922'),
                    name='Lost',
                    probability=0.00,
                    display_order=5
                )
            ]
            db.add_all(stages)
            db.commit()

        # Seed Permissions
        if db.query(models.Permission).first() is None:
            permissions = [
                models.Permission(
                    id=UUID('7a2f6448-ff35-4e08-bfb1-912c40c83a71'),
                    name='convert_leads',
                    description='Convert inbound unqualified leads into active deals'
                ),
                models.Permission(
                    id=UUID('8b3f6448-ff35-4e08-bfb1-912c40c83a72'),
                    name='edit_deals',
                    description='Update valuations and stages on deal entities'
                ),
                models.Permission(
                    id=UUID('9c4f6448-ff35-4e08-bfb1-912c40c83a73'),
                    name='manage_roles',
                    description='Assign and edit operators permissions scopes'
                )
            ]
            db.add_all(permissions)
            db.commit()
    except Exception as e:
        print(f"Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()

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
