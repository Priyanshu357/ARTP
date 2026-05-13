"""LLM-powered enhancement for diagnostic explanations.

This module provides optional LLM integration to generate more sophisticated
and context-aware explanations for critical diagnostic issues.

Features:
- Support for OpenAI and Anthropic APIs
- Support for GitHub Models (free via GitHub Marketplace)
- In-memory caching to avoid duplicate API calls
- Graceful fallback when API unavailable
- Only enhances CRITICAL/HIGH severity diagnostics (cost optimization)
"""

import os
import json
from typing import Dict, Optional, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LLMEnhancer:
    """Enhance diagnostics with LLM-generated explanations.

    This class integrates

 with OpenAI or Anthropic APIs to provide deeper
    technical insights for critical diagnostic issues. It includes automatic
    caching and graceful fallback if the API is unavailable.

    Example:
        enhancer = LLMEnhancer(provider="openai")
        if enhancer.enabled:
            enhanced = enhancer.enhance_diagnostic(diagnostic, context)
    """

    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        """Initialize LLM enhancer.

        Args:
            provider: "openai", "anthropic", or "github"
            api_key: API key (if None, reads from environment variable)
        """
        self.provider = provider
        self.api_key = api_key or self._get_api_key_from_env()
        self.enabled = self.api_key is not None
        self._cache = {}  # Simple in-memory cache

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variable."""
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        elif self.provider == "github":
            return os.getenv("GITHUB_TOKEN")
        else:
            return None

    def enhance_diagnostic(self, diagnostic: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a diagnostic with LLM-generated content.

        Args:
            diagnostic: Diagnostic dict with severity, description, evidence, etc.
            context: Context dict with model_name, task, attacks, num_samples

        Returns:
            Enhanced diagnostic dict with llm_enhanced=True or unchanged if LLM unavailable
        """
        if not self.enabled:
            return diagnostic  # Graceful fallback

        # Check cache first
        cache_key = self._make_cache_key(diagnostic)
        if cache_key in self._cache:
            return self._apply_cached_enhancements(diagnostic, self._cache[cache_key])

        try:
            # Generate enhancements via LLM API
            prompt = self._build_prompt(diagnostic, context)
            response = self._call_api(prompt)
            enhancements = self._parse_response(response)

            # Cache for future use
            self._cache[cache_key] = enhancements

            # Apply enhancements to diagnostic
            return self._apply_enhancements(diagnostic, enhancements)

        except Exception as e:
            print(f"[WARNING] LLM enhancement failed: {e}")
            return diagnostic  # Graceful fallback

    def _make_cache_key(self, diagnostic: Dict[str, Any]) -> str:
        """Create cache key from diagnostic signature.

        Args:
            diagnostic: Diagnostic dict

        Returns:
            String cache key based on diagnostic type and key evidence
        """
        key_parts = [
            diagnostic.get('diagnostic', ''),
            diagnostic.get('severity', ''),
            str(diagnostic.get('evidence', {}).get('dominant_class', '')),
            str(diagnostic.get('evidence', {}).get('mean_confidence', ''))[:10]  # Truncate float
        ]
        return "|".join(key_parts)

    def _apply_cached_enhancements(self, diagnostic: Dict[str, Any], enhancements: Dict[str, Any]) -> Dict[str, Any]:
        """Apply cached enhancements to diagnostic.

        Args:
            diagnostic: Original diagnostic dict
            enhancements: Cached enhancements dict

        Returns:
            Enhanced diagnostic dict
        """
        return self._apply_enhancements(diagnostic.copy(), enhancements)

    def _build_prompt(self, diagnostic: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build structured prompt for LLM analysis.

        Args:
            diagnostic: Diagnostic dict
            context: Context dict with model info

        Returns:
            Formatted prompt string
        """
        return f"""You are an AI security expert analyzing ML model robustness test results.

**Issue Detected:**
{diagnostic.get('description', 'Unknown issue')}

**Severity:** {diagnostic.get('severity', 'UNKNOWN')}

**Evidence:**
{json.dumps(diagnostic.get('evidence', {}), indent=2)}

**Model Context:**
- Model: {context.get('model_name', 'Unknown')}
- Task: {context.get('task', 'Text Classification')}
- Attacks: {', '.join(context.get('attacks', []))}
- Samples: {context.get('num_samples', 'Unknown')}

**Current Analysis:**
- Root Causes: {diagnostic.get('root_causes', [])}
- Recommendations: {diagnostic.get('recommendations', [])}

**Task:**
Provide enhanced analysis with:
1. **Enhanced Explanation** (2-4 sentences): Deep technical insight into WHY this is happening
2. **Additional Root Causes** (2-3 items): Causes not already listed above
3. **Priority Recommendations** (2-3 items): Most critical next steps to address this issue

Respond in JSON format:
{{
  "enhanced_explanation": "...",
  "additional_root_causes": ["cause1", "cause2"],
  "priority_recommendations": ["rec1", "rec2"]
}}"""

    def _call_api(self, prompt: str) -> str:
        """Call LLM API based on provider.

        Args:
            prompt: Prompt string

        Returns:
            API response text

        Raises:
            ValueError: If provider is unknown
            Exception: If API call fails
        """
        if self.provider == "openai":
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Fast, cost-effective
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,  # Lower temp for consistent analysis
                max_tokens=800
            )
            return response.choices[0].message.content

        elif self.provider == "github":
            import openai
            # GitHub Models API uses OpenAI client with custom endpoint
            # Reads GITHUB_TOKEN from .env file
            client = openai.OpenAI(
                base_url="https://models.github.ai/inference",
                api_key=self.api_key
            )
            response = client.chat.completions.create(
                model="openai/gpt-5",  # GitHub Marketplace model
                messages=[
                    {"role": "system", "content": "You are an AI security expert analyzing ML model robustness."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                # GitHub Models only supports default temperature (1)
                max_completion_tokens=800  # GitHub uses max_completion_tokens, not max_tokens
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=800,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM.

        Args:
            response: Response string from API

        Returns:
            Parsed enhancements dict
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback: return empty enhancements if parsing fails
            print("[WARNING] Failed to parse LLM response as JSON")
            return {
                "enhanced_explanation": response[:500],  # Truncate if too long
                "additional_root_causes": [],
                "priority_recommendations": []
            }

    def _apply_enhancements(self, diagnostic: Dict[str, Any], enhancements: Dict[str, Any]) -> Dict[str, Any]:
        """Merge LLM enhancements into diagnostic.

        Args:
            diagnostic: Original diagnostic dict
            enhancements: Enhancements dict from LLM

        Returns:
            Enhanced diagnostic dict with llm_enhanced flag
        """
        # Update interpretation with enhanced explanation
        if enhancements.get('enhanced_explanation'):
            diagnostic['interpretation'] = enhancements['enhanced_explanation']

        # Append additional root causes (don't replace existing)
        if enhancements.get('additional_root_causes'):
            diagnostic['root_causes'] = (
                diagnostic.get('root_causes', []) +
                enhancements['additional_root_causes']
            )

        # Append priority recommendations (don't replace existing)
        if enhancements.get('priority_recommendations'):
            diagnostic['recommendations'] = (
                diagnostic.get('recommendations', []) +
                enhancements['priority_recommendations']
            )

        # Mark as LLM-enhanced
        diagnostic['llm_enhanced'] = True

        return diagnostic
