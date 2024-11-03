from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union

# Base FHIR Models
class Quantity(BaseModel):
    value: Optional[float] = None 
    comparator: Optional[Literal["<", "<=", ">=", ">", "ad"]] = None
    unit: Optional[str] = None 
    system: Optional[str] = None 
    code: Optional[str] = None

class Coding(BaseModel):
    system: str
    code: str
    display: Optional[str] = None

class CodeableConcept(BaseModel):
    coding: List[Coding] = Field(default_factory=list)
    text: Optional[str] = None

class Identifier(BaseModel):
    system: Optional[str] = None
    value: str

class HumanName(BaseModel):
    text: Optional[str] = None
    family: Optional[str] = None
    given: Optional[List[str]] = Field(default_factory=list)
    prefix: Optional[List[str]] = Field(default_factory=list)
    suffix: Optional[List[str]] = Field(default_factory=list)

class Address(BaseModel):
    text: Optional[str] = None
    line: Optional[List[str]] = Field(default_factory=list)
    city: Optional[str] = None
    state: Optional[str] = None
    postalCode: Optional[str] = None
    country: Optional[str] = None

class Dosage(BaseModel):
    text: Optional[str] = None
    timing: Optional[CodeableConcept] = None
    route: Optional[CodeableConcept] = None
    method: Optional[CodeableConcept] = None

# FHIR Resources
class Patient(BaseModel):
    resourceType: Literal["Patient"] = "Patient"
    identifier: Optional[List[Identifier]] = Field(default_factory=list)
    name: Optional[List[HumanName]] = Field(default_factory=list)
    gender: Optional[Literal["male", "female", "other", "unknown"]] = None
    birthDate: Optional[str] = None
    address: Optional[List[Address]] = Field(default_factory=list)

class Observation(BaseModel):
    resourceType: Literal["Observation"] = "Observation"
    identifier: Optional[List[Identifier]] = Field(default_factory=list)
    status: Literal["registered", "preliminary", "final", "amended", "corrected", "cancelled", "entered-in-error", "unknown"]
    code: CodeableConcept
    effectiveDateTime: Optional[str] = None
    valueQuantity: Optional[Quantity] = None

class Condition(BaseModel):
    resourceType: Literal["Condition"] = "Condition"
    identifier: Optional[List[Identifier]] = Field(default_factory=list)
    clinicalStatus: CodeableConcept
    code: CodeableConcept
    onsetDateTime: Optional[str] = None

class MedicationStatement(BaseModel):
    resourceType: Literal["MedicationStatement"] = "MedicationStatement"
    identifier: Optional[List[Identifier]] = Field(default_factory=list)
    status: Literal["recorded", "entered-in-error", "draft"]
    medicationCodeableConcept: Optional[CodeableConcept] = None
    effectiveDateTime: Optional[str] = None
    dosage: Optional[List[Dosage]] = Field(default_factory=list)

class Bundle(BaseModel):
    resourceType: Literal["Bundle"] = "Bundle"
    type: Literal["document"]
    timestamp: str
    coding: List[Coding] = Field(default_factory=list)
    entry: List[Union[Patient, Observation, Condition, MedicationStatement]] = Field(default_factory=list) 