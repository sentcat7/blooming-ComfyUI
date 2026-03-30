from pydantic import BaseModel, Field


class PhotaGenerateRequest(BaseModel):
    prompt: str = Field(...)
    num_output_images: int = Field(1)
    aspect_ratio: str = Field(...)
    resolution: str = Field(...)
    profile_ids: list[str] | None = Field(None)


class PhotaEditRequest(BaseModel):
    prompt: str = Field(...)
    images: list[str] = Field(...)
    num_output_images: int = Field(1)
    aspect_ratio: str = Field(...)
    resolution: str = Field(...)
    profile_ids: list[str] | None = Field(None)


class PhotaEnhanceRequest(BaseModel):
    image: str = Field(...)
    num_output_images: int = Field(1)


class PhotaKnownGeneratedSubjectCounts(BaseModel):
    counts: dict[str, int] = Field(default_factory=dict)


class PhotoStudioResponse(BaseModel):
    images: list[str] = Field(..., description="Base64-encoded PNG output images.")
    known_subjects: PhotaKnownGeneratedSubjectCounts = Field(default_factory=PhotaKnownGeneratedSubjectCounts)


class PhotaAddProfileRequest(BaseModel):
    image_urls: list[str] = Field(...)


class PhotaAddProfileResponse(BaseModel):
    profile_id: str = Field(...)


class PhotaProfileStatusResponse(BaseModel):
    profile_id: str = Field(...)
    status: str = Field(
        ...,
        description="Current profile status: VALIDATING, QUEUING, IN_PROGRESS, READY, ERROR, or INACTIVE.",
    )
    message: str | None = Field(default=None, description="Optional error or status message.")
