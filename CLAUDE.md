# CLAUDE.md - Research Project Configuration

## PROJECT OVERVIEW
**Research Focus**: Comparative Analysis of 1D Zigzag Theory vs 2D FEM for Notched Beam Analysis with Deep Learning Integration

### Core Research Objectives

**Objective 1a**: Development of 1D Zigzag Theory Model for Homogeneous Beam with Rectangular Notch
- Implementation of zigzag displacement field theory for homogeneous material systems
- Incorporation of rectangular notch geometry within 1D analytical framework
- Validation of non-standard application of zigzag theory to single-material beam structures

**Objective 1b**: Development of 2D Elastic Finite Element Model with Identical Configuration
- Creation of high-fidelity 2D FEM model matching exact beam and notch specifications
- Implementation of identical geometric parameters and boundary conditions
- Establishment of reference solution for comparative analysis

**Objective 2**: Multi-Fidelity Surrogate Model Development through Deep Learning Architecture
- **Low Fidelity Surrogate Model (LFSM)**: Conditional AutoEncoder training on 1D zigzag responses (750 cases × 11 sensor points)
- **Latent Space Prediction**: XGBoost implementation for parameter-to-latent space mapping
- **Response Reconstruction**: Direct parameter-to-response prediction via XGBoost-Decoder pipeline
- **Multi-Fidelity Enhancement**: LFSM fine-tuning using limited 2D FEM data (50 cases × 11 sensors) to create Multi-Fidelity Surrogate Model (MFSM)
- **Comparative Validation**: MFSM performance assessment against High Fidelity Surrogate Model (HFSM) trained exclusively on 2D data

**Objective 3**: Inverse Problem Solution using MFSM as Forward Solver
- **Parameter Identification**: Determination of unknown notch parameters (location, depth, width) from measured response data
- **Optimization Strategy**: Differential evolution algorithm implementation with strategic population initialization
- **Search Space Management**: 15 individuals initialized within high-confidence regions, 10 at periphery boundaries, 5 in extended parameter space
- **Training Data Integration**: 2D training dataset utilization for search region refinement and confidence assessment

**Special Note**: 1D zigzag theory typically applies to composite materials but is being applied to homogeneous beam with notch in this study (non-standard application)

## CRITICAL CONSTRAINTS & REQUIREMENTS

### Writing Standards
- **MANDATORY**: Extensive online research for every technical concept, methodology, and related work
- **NO FABRICATION POLICY**: Never create, assume, or fabricate technical details, research findings, equations, or methodological information
- **Academic Tone**: Maintain strict formal academic writing throughout
- **Human-like Writing**: Remove AI-detection patterns while preserving technical accuracy
- **Comprehensive Coverage**: Provide exhaustive coverage with detailed explanations
- **Source Verification**: Cross-reference all claims with credible academic sources

### Technical Specifications
- **1D Zigzag Theory**: Applied to homogeneous notched beam (non-standard application)
- **2D FEM Modeling**: Standard approach for comparative analysis
- **Deep Learning**: Architecture selection through research (U-NET candidate)
- **Crack Parameter Prediction**: Primary deliverable requirement

## FILE STRUCTURE & NAVIGATION

### Code Locations

- **All code files** are stored in the `Code/` directory at the repository root. Use this folder as the canonical source for scripts used in data generation, model training, and evaluation.
- **Code descriptions and documentation** for individual scripts are collected in `Code/codedescription/`. Consult these markdown files before modifying or running any script; they contain usage notes, input/output formats, and expected dependencies.
- **Main LaTeX file**: The project thesis manuscript is `main.tex` at the repository root. IMPORTANT: Do not edit `main.tex` directly unless explicitly instructed. Instead, add instructions or changes in a separate document and request approval before modifying `main.tex`.

When running or modifying code, prefer to work inside the `Code/` directory and ensure any generated outputs follow the project's Output File Management rules (all outputs must be placed in `Claude_res/`).

### Key Reference Files
- **`/Nomenclature.md/` file**: Contains complete notation and symbol definitions
  - **INSTRUCTION**: Always consult nomenclature file before using any symbols
  - **NEW NOMENCLATURE**: If required symbols not present, add them to the file
  - **CONSISTENCY**: Use only approved nomenclature throughout project

### Document Structure
- **Synopsis Report**: Follow reference PDF structural format
- **Chapter-by-Chapter Approach**: Work systematically, request approval before proceeding
- **Literature Integration**: Incorporate extensive literature review in each section

## RESEARCH & VERIFICATION PROTOCOL

### Mandatory Research Steps
1. **Web Search**: Conduct thorough online research for all technical concepts
2. **Source Verification**: Use multiple credible academic sources
3. **Cross-Reference**: Validate all technical claims
4. **Gap Identification**: Explicitly state when information is unavailable
5. **Current State-of-Art**: Include recent developments and findings

### Information Handling
- **Unknown Information**: Request clarification rather than fabricating
- **Technical Definitions**: Ensure accurate explanation of all terminology
- **Methodological Details**: Provide comprehensive coverage of approaches
- **Limitations**: Clearly state research boundaries and constraints

## INTERACTION PROTOCOL

### Communication Guidelines
- **Research First**: Search extensively before responding
- **Complete Sections**: Provide full content without leaving user completions
- **Approval Seeking**: Ask permission before adding supplementary content
- **Focused Responses**: Address only requested content without unnecessary elaboration
- **Clarification Requests**: Ask for specific details when information insufficient

### Content Development Process
1. **Research Phase**: Extensive literature review and technical verification
2. **Structure Planning**: Outline section before writing
3. **Content Creation**: Develop comprehensive, research-backed content
4. **Review Phase**: Verify academic standards and technical accuracy
5. **Approval Request**: Seek confirmation before proceeding to next section

## TECHNICAL WRITING SPECIFICATIONS

### Academic Standards
- **Formality Level**: Strict academic tone throughout
- **Technical Precision**: Accurate definition and explanation of all concepts
- **Literature Integration**: Comprehensive review of current research
- **Methodological Rigor**: Detailed coverage of analytical approaches
- **Result Presentation**: Clear, evidence-based findings

### Content Requirements
- **Comprehensive Coverage**: Exhaustive treatment of each topic
- **Technical Depth**: Detailed explanations of complex concepts
- **Research Integration**: Extensive incorporation of literature findings
- **Comparative Analysis**: Clear presentation of methodology differences
- **Predictive Framework**: Detailed coverage of deep learning integration

## DEEP LEARNING ARCHITECTURE CONSIDERATIONS

### Current Status
- **Architecture Selection**: Under research and evaluation
- **Primary Candidates**: U-NET, complex neural networks
- **Application Focus**: Discrepancy analysis and crack parameter prediction
- **Research Required**: Extensive literature review for optimal architecture selection

### Implementation Requirements
- **Comparative Framework**: Integration with 1D/2D analysis results
- **Predictive Capabilities**: Crack parameter estimation
- **Validation Protocol**: Comprehensive testing and verification
- **Performance Metrics**: Accuracy, efficiency, reliability measures

## PROJECT DELIVERABLES

### Primary Outputs
- **Synopsis Report**: Comprehensive research document
- **Comparative Analysis**: 1D zigzag vs 2D FEM methodology assessment
- **Deep Learning Framework**: Predictive model for crack parameters
- **Technical Documentation**: Complete methodological coverage

### Quality Standards
- **Academic Rigor**: Peer-review level quality
- **Technical Accuracy**: Verified through multiple sources
- **Comprehensive Coverage**: Complete treatment of all aspects
- **Original Research**: Novel application of zigzag theory to notched beams

## DEVELOPMENT GUIDELINES

### Output File Management
- **CRITICAL**: Always produce files in `Claude_res/` directory and nowhere else
- **NO EXCEPTIONS**: All outputs, results, analysis files, figures, and generated content MUST go to `Claude_res/`
- **Directory Creation**: Create `Claude_res/` if it doesn't exist before generating any output
- **Path Validation**: Verify output path targets `Claude_res/` before writing files
- **NO UNNECESSARY DOCUMENTATION**: Do not create README, documentation, or explanation files after writing code unless explicitly requested
- **EDIT, DON'T CREATE**: When improvements are needed, edit existing files rather than creating new versions

### Code Development (if required)
- **Default Language**: Python unless specified otherwise
- **Analysis Tool**: JavaScript only in analysis environment
- **Single File Principle**: One cohesive file unless explicitly requested
- **Validation Protocol**: Mandatory syntax and logic checking

### Documentation Standards
- **Technical Documentation**: Complete coverage of all implementations
- **Code Comments**: Comprehensive explanation of algorithms
- **Method Documentation**: Detailed coverage of all approaches
- **Result Documentation**: Clear presentation of findings

## FINAL REMINDERS

### Critical Success Factors
- **OUTPUT LOCATION - MANDATORY**: ALL files MUST be produced in `Claude_res/` directory only
- **Research Thoroughness**: Extensive online investigation required
- **Academic Standards**: Maintain strict formal requirements
- **Technical Accuracy**: Verify all information through credible sources
- **Human-like Writing**: Remove AI detection patterns
- **Comprehensive Coverage**: Complete treatment of all topics

### Collaboration Protocol
- **Step-by-Step Approach**: Work systematically through each section
- **Approval Requirements**: Seek confirmation before major additions
- **Clarification Protocol**: Request details when information insufficient
- **Quality Assurance**: Maintain highest academic and technical standards

---

**Remember**: This research represents a novel application of 1D zigzag theory to homogeneous notched beams, requiring careful documentation of methodological approaches and comprehensive comparative analysis with standard 2D FEM techniques.
- No need to make unneccesary documents after you write a code