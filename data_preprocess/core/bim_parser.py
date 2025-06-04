# data_preprocess/core/bim_parser.py
import ifcopenshell
import ifcopenshell.geom
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np


class IFCComponent:
    def __init__(self, guid: str, ifc_type: str, name: str,
                 vertices: Optional[np.ndarray] = None,
                 faces: Optional[np.ndarray] = None,
                 ifc_element: Optional[ifcopenshell.entity_instance] = None):
        self.guid = guid
        self.ifc_type = ifc_type
        self.name = name  # Component name from IFC
        self.vertices = vertices  # Nx3 numpy array
        self.faces = faces  # MxK numpy array (K depends on face polygon size)
        self.ifc_element = ifc_element  # Store the raw ifc_element for potential advanced queries

    def __repr__(self):
        return f"IFCComponent(guid={self.guid}, type={self.ifc_type}, name='{self.name}', vertices={self.vertices.shape if self.vertices is not None else None})"


class BIMSceneParser:
    def __init__(self, ifc_file_path: Path, target_entities: Optional[List[str]] = None):
        if not ifc_file_path.exists():
            raise FileNotFoundError(f"IFC file not found: {ifc_file_path}")
        try:
            self.ifc_file = ifcopenshell.open(str(ifc_file_path))
        except Exception as e:
            raise IOError(f"Failed to open or parse IFC file {ifc_file_path}: {e}")

        self.target_entities = target_entities if target_entities else []
        # ifcopenshell settings for geometry creation
        self.settings = ifcopenshell.geom.settings()
        self.settings.set(self.settings.USE_WORLD_COORDS, True)  # Use world coordinates
        self.settings.set(self.settings.CONVERT_BACK_UNITS, True)  # Ensure units are consistent (meters)
        self.settings.set(self.settings.WELD_VERTICES, True)  # Weld vertices to reduce redundancy
        # self.settings.set(self.settings.SEW_SHELLS, True)  # Sew shells for solid geometry
        # self.settings.set(self.settings.FASTER_BOOLEANS, True)  # Potentially faster boolean ops, check impact
        self.length_unit_factor = self._get_length_unit_conversion_factor()
        if self.length_unit_factor != 1.0:
            print(
                f"INFO: IFC file {ifc_file_path.name} units require scaling."
                f"Factor: {self.length_unit_factor} (to meters)."
            )

    def _get_length_unit_conversion_factor(self) -> float:
        """
        Determines the conversion factor to meters based on the IFC file's length unit.
        Returns 1.0 if already in meters or unit not found.
        """
        units_assignment = self.ifc_file.by_type("IfcUnitAssignment")
        if not units_assignment:
            print("WARNING: No IfcUnitAssignment found. Assuming meters, but coordinates might be incorrect.")
            return 1.0
        for unit in units_assignment[0].Units:
            if unit.is_a("IfcSIUnit") and hasattr(unit, 'UnitType') and unit.UnitType == "LENGTHUNIT":
                if hasattr(unit, 'Name'):
                    if unit.Name == "METRE": return 1.0
                    if unit.Name == "MILLIMETRE":
                        if hasattr(unit, 'Prefix') and unit.Prefix == "MILLI": return 0.001
                        return 0.001
                    if unit.Name == "CENTIMETRE": return 0.01
                    if unit.Name == "FOOT": return 0.3048
                    if unit.Name == "INCH": return 0.0254
                if hasattr(unit, 'Prefix'):
                    if unit.Prefix == "MILLI" and unit.Name == "METRE":return 0.001
                    if unit.Prefix == "CENTI" and unit.Name == "METRE":return 0.01
            elif unit.is_a("IfcConversionBasedUnit") and hasattr(unit, 'UnitType') and unit.UnitType == "LENGTHUNIT":
                # This requires finding the IfcMeasureWithUnit and its ValueComponent (an IfcReal)
                # and the ConversionFactor which itself is an IfcMeasureWithUnit.
                # This can get complex. For now, we'll assume IfcSIUnit is primary.
                print(f"WARNING: Encountered IfcConversionBasedUnit for LENGTHUNIT for {self.ifc_file.filename}. This is not fully handled yet. Assuming meters.")
                if hasattr(unit, 'ConversionFactor') and unit.ConversionFactor is not None:
                    # This is a simplification. The ConversionFactor itself has a unit.
                    # We'd need to recursively resolve to get the factor relative to SI meters.
                    # For now, if we find a ValueComponent, we might make an assumption.
                    if hasattr(unit.ConversionFactor, 'ValueComponent') and \
                            isinstance(unit.ConversionFactor.ValueComponent, ifcopenshell.entity_instance) and \
                            unit.ConversionFactor.ValueComponent.is_a('IfcReal'):
                        # This is highly speculative without knowing the base unit of the conversion factor
                        # For example, if it says 1000 and the base is mm converting to m, this is wrong.
                        print(f"Found conversion factor value: {unit.ConversionFactor.ValueComponent.wrappedValue}")
                        pass
        print(f"WARNING: Could not reliably determine length unit for {self.ifc_file.filename}. Assuming meters. Please verify coordinates.")
        return 1.0

    def get_components(self) -> List[IFCComponent]:
        components: List[IFCComponent] = []

        # If target_entities is empty, process all IfcProduct types
        # Otherwise, filter by the specified entity types
        ifc_products_to_process = []
        if not self.target_entities:
            ifc_products_to_process = self.ifc_file.by_type("IfcProduct")
        else:
            for entity_type_str in self.target_entities:
                try:
                    ifc_products_to_process.extend(self.ifc_file.by_type(entity_type_str))
                except Exception as e:
                    print(f"Warning: Could not query IFC type '{entity_type_str}': {e}")
        print(f"Found {len(ifc_products_to_process)} products of targeted types to process.")
        for element in ifc_products_to_process:
            if not element.Representation:  # Skip elements without geometric representation
                continue
            try:
                shape = ifcopenshell.geom.create_shape(self.settings, element)
                # shape.geometry is an OpenCASCADE TopoDS_Shape object
                # We need to convert this to vertices and faces
                # ifcopenshell.geom.tesselate can convert TopoDS_Shape to vertices and faces
                # The result format is: verts (tuple of floats), faces (tuple of ints)
                # verts: (v1x, v1y, v1z, v2x, v2y, v2z, ...)
                # faces: (f1v1, f1v2, f1v3, f2v1, f2v2, f2v3, ...) for triangles

                # Note: ifcopenshell.geom.tesselate is often preferred over shape.geometry.verts/faces
                # as it handles more complex geometry and BREP representations better.
                # However, the exact API for direct tesselation from TopoDS_Shape might vary
                # or require more steps.
                # A common approach is to use PythonOCC or similar to process TopoDS_Shape.
                # For simplicity, if `shape.geometry.verts` is directly available and suitable, we use it.
                # Otherwise, more complex BREP processing would be needed here.

                if hasattr(shape.geometry, 'verts') and hasattr(shape.geometry, 'faces'):
                    verts_flat = np.array(shape.geometry.verts, dtype=np.float32)
                    faces_flat = np.array(shape.geometry.faces, dtype=np.int32)

                    if not verts_flat.size or not faces_flat.size:
                        # print(f"Warning: Element {element.GlobalId} ({element.is_a()}) has empty geometry (verts/faces). Skipping.")
                        continue

                    vertices = verts_flat.reshape(-1, 3)

                    # Assuming faces are triangles. If not, they need to be triangulated.
                    # For many IFC elements, the default tessellation is often triangles.
                    num_face_verts = 3  # Assuming triangular faces from ifcopenshell's default tessellation
                    if faces_flat.shape[0] % num_face_verts != 0:
                        # print(f"Warning: Non-triangular faces for {element.GlobalId} ({element.is_a()}). This might need triangulation. Skipping for now.")
                        # A more robust solution would involve triangulating these faces.
                        # For now, we'll try to infer face size if possible, or skip.
                        # This is a simplification; robust handling of arbitrary polygons is complex.
                        is_polygon = True
                        temp_faces = []
                        current_face = []
                        # Try to infer face structure if it's not simple triangles.
                        # This part is heuristic and might not cover all IFC cases.
                        # A safer bet is to ensure IFC files are exported with triangulated meshes
                        # or use a robust BRep to mesh library.
                        # For now, let's assume ifcopenshell provides triangle indices.
                        # If not, this will likely fail or produce incorrect faces.
                        # If the number of indices per face is variable, this simple reshape won't work.
                        # We should check if faces_flat can be reshaped to (N, 3)
                        try:
                            faces = faces_flat.reshape(-1, 3)
                        except ValueError:
                            print(f"Could not reshape faces for {element.GlobalId}. Face data: {faces_flat[:10]}")
                            continue  # Skip this element
                    else:
                        faces = faces_flat.reshape(-1, 3)
                    component = IFCComponent(
                        guid=element.GlobalId,
                        ifc_type=element.is_a(),
                        name=element.Name if element.Name else "Unnamed",
                        vertices=vertices,
                        faces=faces,
                        ifc_element=element
                    )
                    components.append(component)
                else:
                    print(f"Warning: Element {element.GlobalId} ({element.is_a()}) has geometry but no direct verts/faces attributes. Skipping.")
                    pass
            except RuntimeError as e:
                print(f"Warning: Could not create shape for element {element.GlobalId} ({element.is_a()}): {e}. Skipping.")
                pass
            except Exception as e:
                print(f"An unexpected error occurred while processing element {element.GlobalId} ({element.is_a()}): {e}. Skipping.")
                pass

        print(f"Successfully parsed {len(components)} components with geometry.")
        return components

    def get_project_length_unit(self) -> Optional[str]:
        """Tries to find the project's length unit."""
        units = self.ifc_file.by_type("IfcUnitAssignment")
        if units:
            for unit in units[0].Units:
                if unit.is_a("IfcSIUnit") and unit.UnitType == "LENGTHUNIT":
                    return unit.Name
        return None

    def get_project_length_unit(self) -> Optional[str]:
        """Tries to find the project's length unit from IfcUnitAssignment."""
        unit_assignments = self.ifc_file.by_type("IfcUnitAssignment")
        if not unit_assignments:
            return None
        for unit_assignment in unit_assignments:
            if not hasattr(unit_assignment, 'Units') or not unit_assignment.Units:
                continue
            for unit in unit_assignment.Units:
                if unit.is_a("IfcSIUnit") and hasattr(unit, 'UnitType') and unit.UnitType == "LENGTHUNIT":
                    # Prefer Name attribute for the unit (e.g., METRE, MILLIMETRE)
                    if hasattr(unit, 'Name') and unit.Name:
                        return unit.Name
                        # Fallback for some IFC versions or if Name is not set, try to infer from Prefix
                    # This is less reliable as it requires knowing the base unit (usually meter)
                    elif hasattr(unit, 'Prefix') and unit.Prefix:
                        # If Prefix is MILLI, and base is METER, then it's MILLIMETRE
                        # If no explicit base unit is found, this part can be ambiguous.
                        # For simplicity, if only Prefix is available, return it.
                        # A more robust system would check IfcNamedUnit for the base SI unit.
                        # print(f"Debug: Found length unit with Prefix: {unit.Prefix}")
                        return unit.Prefix  # e.g. "MILLI"
        return None  # Default or if not found