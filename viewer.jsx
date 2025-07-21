import React, { useState, useMemo } from 'react';
import { Card } from "@/components/ui/card";

// --- NEW ---
// 1. Helper function to estimate the size needed for the text.
function getTextDimensions(text) {
    const lines = text.split(' '); // Simple word split
    const charWidth = 6; // Estimated width of a character in pixels
    const lineHeight = 12; // Height of a line in pixels
    const padding = 10; // Padding around the text
    const maxWidth = 90; // Max width of a node before wrapping

    let currentLineWidth = 0;
    let lineCount = 1;

    lines.forEach(word => {
        const wordWidth = word.length * charWidth;
        if (currentLineWidth + wordWidth > maxWidth) {
            lineCount++;
            currentLineWidth = wordWidth;
        } else {
            currentLineWidth += wordWidth;
        }
    });

    const height = lineCount * lineHeight + padding * 2;
    // For circles, the width and height must be equal (diameter)
    const diameter = Math.sqrt(Math.max(currentLineWidth, maxWidth) ** 2 + height ** 2);

    return {
        rectWidth: maxWidth + padding,
        rectHeight: height,
        circleRadius: diameter / 2 + padding / 2,
    };
}


// 2. Updated layout function to handle dynamic node sizes.
function calculateLayout(nodes, links, width) {
    const nodeMap = new Map(nodes.map(n => {
        const dimensions = getTextDimensions(n.id);
        return [n.id, { ...n, children: [], parents: [], dimensions }];
    }));

    links.forEach(link => {
        const sourceNode = nodeMap.get(link.source);
        const targetNode = nodeMap.get(link.target);
        if (sourceNode && targetNode) {
            sourceNode.children.push(targetNode.id);
            targetNode.parents.push(sourceNode.id);
        }
    });

    const layers = [];
    let currentLayerNodes = Array.from(nodeMap.values()).filter(n => n.parents.length === 0);

    while (currentLayerNodes.length > 0) {
        layers.push(currentLayerNodes.map(n => n.id));
        const nextLayerNodeIds = new Set();
        currentLayerNodes.forEach(node => {
            node.children.forEach(childId => nextLayerNodeIds.add(childId));
        });
        currentLayerNodes = Array.from(nextLayerNodeIds).map(id => nodeMap.get(id));
    }

    const nodePositions = new Map();
    let currentY = 50;

    layers.forEach((layer, i) => {
        const xPadding = 80;
        const xStep = (width - xPadding * 2) / (layer.length - 1 || 1);
        let maxLayerHeight = 0;

        layer.forEach((nodeId, j) => {
            const node = nodeMap.get(nodeId);
            const nodeHeight = node.shape === 'rect' ? node.dimensions.rectHeight : node.dimensions.circleRadius * 2;
            maxLayerHeight = Math.max(maxLayerHeight, nodeHeight);
            
            nodePositions.set(nodeId, {
                x: layer.length === 1 ? width / 2 : xPadding + j * xStep,
                y: currentY + nodeHeight / 2,
            });
        });
        currentY += maxLayerHeight + 50; // Update Y for the next layer
    });

    const totalHeight = currentY;
    return { nodePositions, totalHeight };
}


export default function FlowChartViewer() {
    const { nodes, links } = props.data || { nodes: [], links: [] };
    const [tooltip, setTooltip] = useState(null);

    const { nodePositions, totalHeight } = useMemo(() => {
        const nodeDetails = nodes.map(n => ({...n, dimensions: getTextDimensions(n.id)}));
        return calculateLayout(nodeDetails, links, 800);
    }, [nodes, links]);

    const handleMouseOver = (event, node) => {
        setTooltip({ content: node.attributes, x: event.clientX, y: event.clientY });
    };

    const handleMouseOut = () => { setTooltip(null); };
    const handleClick = (node) => { callAction({ name: "node_clicked", payload: node }); };

    return (
        <Card className="p-2 relative overflow-auto" style={{ width: '800px', height: '600px' }}>
            <svg width="800" height={totalHeight}>
                <g>
                    {links.map((link, i) => {
                        const sourceNode = nodes.find(n => n.id === link.source);
                        const targetNode = nodes.find(n => n.id === link.target);
                        const sourcePos = nodePositions.get(link.source);
                        const targetPos = nodePositions.get(link.target);
                        if (!sourcePos || !targetPos || !sourceNode || !targetNode) return null;

                        const gap = targetNode.shape === 'rect' ? targetNode.dimensions.rectHeight / 2 : targetNode.dimensions.circleRadius;
                        const dx = targetPos.x - sourcePos.x;
                        const dy = targetPos.y - sourcePos.y;
                        const length = Math.sqrt(dx * dx + dy * dy);
                        const newTargetX = targetPos.x - (dx / length) * gap;
                        const newTargetY = targetPos.y - (dy / length) * gap;

                        return (
                            <path key={i} d={`M${sourcePos.x},${sourcePos.y}L${newTargetX},${newTargetY}`} stroke="#999" strokeWidth="2" fill="none" markerEnd="url(#arrow)" />
                        );
                    })}
                </g>
                <g>
                    {nodes.map(node => {
                        const pos = nodePositions.get(node.id);
                        if (!pos) return null;
                        const { rectWidth, rectHeight, circleRadius } = node.dimensions;
                        const commonProps = {
                            onMouseOver: (e) => handleMouseOver(e, node), onMouseOut: handleMouseOut,
                            onClick: () => handleClick(node), style: { cursor: 'pointer' }, fill: node.color || '#999'
                        };

                        return (
                            // --- NEW ---
                            // 3. Render SVG elements using the calculated dynamic sizes.
                            <g key={node.id} transform={`translate(${pos.x}, ${pos.y})`}>
                                {node.shape === 'rect' ? (
                                    <rect x={-rectWidth / 2} y={-rectHeight / 2} width={rectWidth} height={rectHeight} rx="5" {...commonProps} />
                                ) : (
                                    <circle r={circleRadius} {...commonProps} />
                                )}
                                <foreignObject x={-rectWidth / 2} y={-rectHeight / 2} width={rectWidth} height={rectHeight} style={{pointerEvents: 'none'}}>
                                   <div style={{
                                       width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                                       textAlign: 'center', color: 'white', fontSize: '10px', fontWeight: 'bold',
                                       fontFamily: 'sans-serif', wordBreak: 'break-word', padding: '5px'
                                   }}>
                                       {node.id}
                                   </div>
                                </foreignObject>
                            </g>
                        );
                    })}
                </g>
                <defs>
                    <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#999" />
                    </marker>
                </defs>
            </svg>
            {tooltip && ( /* Tooltip JSX remains the same */ )}
        </Card>
    );
}
