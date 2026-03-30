import base64
from io import BytesIO

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.phota_labs import (
    PhotaAddProfileRequest,
    PhotaAddProfileResponse,
    PhotaEditRequest,
    PhotaEnhanceRequest,
    PhotaGenerateRequest,
    PhotaProfileStatusResponse,
    PhotoStudioResponse,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    bytesio_to_image_tensor,
    poll_op,
    sync_op,
    upload_images_to_comfyapi,
    upload_image_to_comfyapi,
    validate_string,
)

# Direct API endpoint (comment out this class to use proxy)
class ApiEndpoint(ApiEndpoint):
    """Temporary override to use direct API instead of proxy."""

    def __init__(
        self,
        path: str,
        method: str = "GET",
        *,
        query_params: dict | None = None,
        headers: dict | None = None,
    ):
        self.path = path.replace("/proxy/phota/", "https://api.photalabs.com/")
        self.method = method
        self.query_params = query_params or {}
        self.headers = headers or {}
        if "api.photalabs.com" in self.path:
            self.headers["X-API-Key"] = "YOUR_PHOTA_API_KEY"


PHOTA_LABS_PROFILE_ID = "PHOTA_LABS_PROFILE_ID"


class PhotaLabsGenerate(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PhotaLabsGenerate",
            display_name="Phota Labs Generate",
            category="api node/image/Phota Labs",
            description="Generate images from a text prompt using Phota Labs.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt describing the desired image.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["auto", "1:1", "3:4", "4:3", "9:16", "16:9"],
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["1K", "4K"],
                ),
            ],
            outputs=[IO.Image.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        aspect_ratio: str,
        resolution: str,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False, min_length=1)
        pid_list = None  # list(profile_ids.values()) if profile_ids else None
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/phota/v1/phota/generate", method="POST"),
            response_model=PhotoStudioResponse,
            data=PhotaGenerateRequest(
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                profile_ids=pid_list or None,
            ),
        )
        return IO.NodeOutput(bytesio_to_image_tensor(BytesIO(base64.b64decode(response.images[0]))))


class PhotaLabsEdit(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PhotaLabsEdit",
            display_name="Phota Labs Edit",
            category="api node/image/Phota Labs",
            description="Edit images based on a text prompt using Phota Labs. "
            "Provide input images and a prompt describing the desired edit.",
            inputs=[
                IO.Autogrow.Input(
                    "images",
                    template=IO.Autogrow.TemplatePrefix(
                        IO.Image.Input("image"),
                        prefix="image",
                        min=1,
                        max=10,
                    ),
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["auto", "1:1", "3:4", "4:3", "9:16", "16:9"],
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["1K", "4K"],
                ),
                IO.Autogrow.Input(
                    "profile_ids",
                    template=IO.Autogrow.TemplatePrefix(
                        IO.Custom(PHOTA_LABS_PROFILE_ID).Input("profile_id"),
                        prefix="profile_id",
                        min=0,
                        max=5,
                    ),
                ),
            ],
            outputs=[IO.Image.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        images: IO.Autogrow.Type,
        prompt: str,
        aspect_ratio: str,
        resolution: str,
        profile_ids: IO.Autogrow.Type = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False, min_length=1)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/phota/v1/phota/edit", method="POST"),
            response_model=PhotoStudioResponse,
            data=PhotaEditRequest(
                prompt=prompt,
                images=await upload_images_to_comfyapi(cls, list(images.values()), max_images=10),
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                profile_ids=list(profile_ids.values()) if profile_ids else None,
            ),
        )
        return IO.NodeOutput(bytesio_to_image_tensor(BytesIO(base64.b64decode(response.images[0]))))


class PhotaLabsEnhance(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PhotaLabsEnhance",
            display_name="Phota Labs Enhance",
            category="api node/image/Phota Labs",
            description="Automatically enhance a photo using Phota Labs. "
            "No text prompt is required — enhancement parameters are inferred automatically.",
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="Input image to enhance.",
                ),
                IO.Autogrow.Input(
                    "profile_ids",
                    template=IO.Autogrow.TemplatePrefix(
                        IO.Custom(PHOTA_LABS_PROFILE_ID).Input("profile_id"),
                        prefix="profile_id",
                        min=0,
                        max=5,
                    ),
                ),
            ],
            outputs=[IO.Image.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
            profile_ids: IO.Autogrow.Type = None,
    ) -> IO.NodeOutput:
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/phota/v1/phota/enhance", method="POST"),
            response_model=PhotoStudioResponse,
            data=PhotaEnhanceRequest(
                image=await upload_image_to_comfyapi(cls, image),
            ),
        )
        return IO.NodeOutput(bytesio_to_image_tensor(BytesIO(base64.b64decode(response.images[0]))))


class PhotaLabsSelectProfile(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PhotaLabsSelectProfile",
            display_name="Phota Labs Select Profile",
            category="api node/image/Phota Labs",
            description="Select a trained Phota Labs profile for use in generation.",
            inputs=[
                IO.Combo.Input(
                    "profile_id",
                    options=[],
                    remote=IO.RemoteOptions(
                        route="http://localhost:9000/phota/profiles",
                        refresh_button=True,
                        item_schema=IO.RemoteItemSchema(
                            value_field="profile_id",
                            label_field="profile_id",
                            preview_url_field="preview_url",
                            preview_type="image",
                        ),
                    ),
                ),
            ],
            outputs=[IO.Custom(PHOTA_LABS_PROFILE_ID).Output(display_name="profile_id")],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(cls, profile_id: str) -> IO.NodeOutput:
        return IO.NodeOutput(profile_id)


class PhotaLabsAddProfile(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PhotaLabsAddProfile",
            display_name="Phota Labs Add Profile",
            category="api node/image/Phota Labs",
            description="Create a training profile from 30-50 reference images using Phota Labs. "
            "Uploads images and starts asynchronous training, returning the profile ID once training is queued.",
            inputs=[
                IO.Autogrow.Input(
                    "images",
                    template=IO.Autogrow.TemplatePrefix(
                        IO.Image.Input("image"),
                        prefix="image",
                        min=30,
                        max=50,
                    ),
                ),
            ],
            outputs=[
                IO.Custom(PHOTA_LABS_PROFILE_ID).Output(display_name="profile_id"),
                IO.String.Output(display_name="status"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        images: IO.Autogrow.Type,
    ) -> IO.NodeOutput:
        image_urls = await upload_images_to_comfyapi(
            cls,
            list(images.values()),
            max_images=50,
            wait_label="Uploading training images",
        )
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/phota/v1/phota/profiles/add", method="POST"),
            response_model=PhotaAddProfileResponse,
            data=PhotaAddProfileRequest(image_urls=image_urls),
        )
        # Poll until validation passes and training is queued/in-progress/ready
        status_response = await poll_op(
            cls,
            ApiEndpoint(
                path=f"/proxy/phota/v1/phota/profiles/{response.profile_id}/status"
            ),
            response_model=PhotaProfileStatusResponse,
            status_extractor=lambda r: r.status,
            completed_statuses=["QUEUING", "IN_PROGRESS", "READY"],
            failed_statuses=["ERROR", "INACTIVE"],
        )
        return IO.NodeOutput(response.profile_id, status_response.status)


class PhotaLabsExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            PhotaLabsGenerate,
            PhotaLabsEdit,
            PhotaLabsEnhance,
            PhotaLabsSelectProfile,
            PhotaLabsAddProfile,
        ]


async def comfy_entrypoint() -> PhotaLabsExtension:
    return PhotaLabsExtension()
