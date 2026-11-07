from datetime import timezone as dt_timezone
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(default='healthy', description='Service health status')
    timestamp: str = Field(default_factory=lambda: datetime.now(dt_timezone.utc).isoformat(), description='UTC ISO timestamp')

class ToolItem(BaseModel):
    name: str = Field(default='', description='Tool identifier')
    description: str = Field(default='', description='Tool description')
    input_schema: Dict[str, Any] = Field(default_factory=dict, description='Input JSON schema')

class ToolsResponse(BaseModel):
    tools: List[Dict[str, Any]] = Field(default_factory=list, description='List of available tools')

class ExplainabilityWeights(BaseModel):
    resilience: float = Field(default=0.3)
    alternative_paths: float = Field(default=0.25)
    mesh_connectivity: float = Field(default=0.2)
    coverage: float = Field(default=0.15)
    stability: float = Field(default=0.1)

class ExplainabilityDetail(BaseModel):
    primary_driver: str = Field(default='communication_resilience')
    weights: ExplainabilityWeights = Field(default_factory=ExplainabilityWeights)

class ComputeAgentRadioIntelligenceRequest(BaseModel):
    target_identifier: str = Field(default='regional_network', description='Target identifier or region name')
    query_parameter: str = Field(default='communication_analysis', description='Query analysis parameter')
    lat: float = Field(default=2.27, description='Latitude [-90, 90]')
    lon: float = Field(default=102.55, description='Longitude [-180, 180]')
    window_steps: int = Field(default=12, description='Temporal window steps')

class ComputeAgentRadioIntelligenceResponse(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(dt_timezone.utc).isoformat())
    status: str = Field(default='healthy')
    primary_metric: float = Field(default=89.6, description='Agent Communication Score (ACS)')
    secondary_score: float = Field(default=0.94, description='Communication Resilience Index (CRI)')
    state_index: float = Field(default=84.7, description='Readiness state index')
    classification: str = Field(default='RESILIENT', description='Readiness classification state')
    explainability: ExplainabilityDetail = Field(default_factory=ExplainabilityDetail)

class GetSpatialH3TensorRequest(BaseModel):
    lat: float = Field(default=2.0442, description='Latitude [-90, 90]')
    lon: float = Field(default=102.5689, description='Longitude [-180, 180]')
    resolution: int = Field(default=8, description='H3 resolution [7, 9]')

class GetSpatialH3TensorResponse(BaseModel):
    h3_index: str = Field(default='886520d81dfffff', description='H3 spatial index')
    spatial_cell: str = Field(default='886520d81dfffff', description='Spatial H3 cell identifier')
    density_score: float = Field(default=64.2, description='Calculated mesh density score')
    terrain_state_index: float = Field(default=75.1, description='Terrain communication state index')
    status: str = Field(default='ACTIVE', description='Spatial cell operational status')

class GenericRadioToolRequest(BaseModel):
    target_identifier: str = Field(default='node_cluster_alpha', description='Target network or node identifier')
    lat: float = Field(default=2.27, description='Latitude coordinate')
    lon: float = Field(default=102.55, description='Longitude coordinate')
    altitude_m: float = Field(default=10.0, description='Antenna or agent altitude in meters')
    resolution: int = Field(default=8, description='H3 resolution index')

class GenericRadioToolResponse(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(dt_timezone.utc).isoformat())
    status: str = Field(default='success')
    metric_value: float = Field(default=85.0, description='Computed primary metric value')
    secondary_metric: float = Field(default=0.9, description='Computed secondary metric value')
    classification: str = Field(default='OPERATIONAL', description='State classification')
    details: Dict[str, Any] = Field(default_factory=dict, description='Additional telemetry data')