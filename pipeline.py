# pipeline.py
import torch
from diffusers import AutoPipelineForText2Image, DDIMScheduler
from config import MODEL_ID
from utils import set_random_seed, tkg_noise

class StableDiffusionGenerator:
    def __init__(self, method: str, device: str, seed: int):
        """
        Initialization:
        - method: "gbp" or "tkg"
        - device: the device to use (e.g., "cuda:0")
        - seed: random seed for reproducibility
        """
        self.method = method
        self.use_tkg = (self.method == "tkg")
        self.device = device
        self.seed = seed
        self.pipe = self.initialize_pipeline()

    def initialize_pipeline(self):
        """Initialize the Stable Diffusion pipeline including its scheduler."""
        scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
        pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID,
            scheduler=scheduler,
            safety_checker=None,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(self.device)
        return pipe

    def generate_images(
        self,
        base_prompts: list,
        active_prompts: str,
        negative_prompts: str,
        guidance_scale: float = 7.5,
        latent_size: int = 128,
        steps: int = 50
    ):
        """
        Generate images based on the specified prompts.
        - base_prompts: list of base prompts
        - active_prompts: additional detailed prompts
        - negative_prompts: prompts for elements to exclude
        """
        # Adjust prompts based on the chosen method
        if self.use_tkg:
            prompts = base_prompts
        else:
            gbp = "isolated on a solid green background"
            prompts = [f"{prompt} {gbp}" for prompt in base_prompts]

        generated_images = []
        for prompt in prompts:
            # Reset the random seed each time to ensure reproducibility
            set_random_seed(self.seed)
            latents = torch.randn((1, 4, latent_size, latent_size), device=self.device, dtype=torch.float16)
            if self.use_tkg:
                latents = tkg_noise(latents, self.device)
            with torch.no_grad():
                image = self.pipe(
                    prompt=prompt + active_prompts,
                    negative_prompts=negative_prompts,
                    latents=latents,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                ).images[0]
            generated_images.append(image)
        return prompts, generated_images
