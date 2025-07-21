import React, { useState, useMemo } from 'react';
import { Card } from "@/components/ui/card";

function getTextDimensions(text) {
    const lines = text.split(' ');
    const charWidth = 6;
    const lineHeight = 12;
    const padding = 10;
    const maxWidth = 90;

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
    const diameter = Math.sqrt(Math.max(currentLineWidth, maxWidth) ** 2 + height ** 2);

    return {
        rectWidth: maxWidth + padding,
        rectHeight: height,
        circleRadius: diameter / 2 + padding / 2,
    };
}

function calculateLayout(nodes, links, width) {
    const nodeMap = new Map(nodes.map(n => ({ ...n, children: [], parents: [] })).map(n => [n.id, n]));

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

    layers.forEach((layer) => {
        const xPadding = 80;
        const xStep = (width - xPadding * 2) / (layer.length - 1 || 1);
        let maxLayerHeight = 0;

        // ✨ THE FIX: Add the index 'j' to the forEach loop arguments.
        layer.forEach((nodeId, j) => {
            const node = nodeMap.get(nodeId);
            const nodeHeight = node.shape === 'rect' ? node.dimensions.rectHeight : node.dimensions.circleRadius * 2;
            maxLayerHeight = Math.max(maxLayerHeight, nodeHeight);
            
            nodePositions.set(nodeId, {
                x: layer.length === 1 ? width / 2 : xPadding + j * xStep,
                y: currentY + nodeHeight / 2,
            });
        });
        currentY += maxLayerHeight + 50;
    });

    const totalHeight = currentY;
    return { nodePositions, totalHeight };
}


export default function FlowChartViewer() {
    const { nodes, links } = props.data || { nodes: [], links: [] };
    const [tooltip, setTooltip] = useState(null);

    const nodesWithDimensions = useMemo(() => {
        return nodes.map(n => ({...n, dimensions: getTextDimensions(n.id)}));
    }, [nodes]);

    const { nodePositions, totalHeight } = useMemo(() => {
        return calculateLayout(nodesWithDimensions, links, 800);
    }, [nodesWithDimensions, links]);

    const handleMouseOver = (event, node) => setTooltip({ content: node.attributes, x: event.clientX, y: event.clientY });
    const handleMouseOut = () => setTooltip(null);
    const handleClick = (node) => callAction({ name: "node_clicked", payload: node });
    
    const nodeMap = useMemo(() => new Map(nodesWithDimensions.map(n => [n.id, n])), [nodesWithDimensions]);

    return (
        <Card className="p-2 relative overflow-auto" style={{ width: '800px', height: '600px' }}>
            <svg width="800" height={totalHeight}>
                {/* Render Nodes First */}
                <g>
                    {nodesWithDimensions.map(node => {
                        const pos = nodePositions.get(node.id);
                        if (!pos) return null;
                        const { rectWidth, rectHeight, circleRadius } = node.dimensions;
                        const commonProps = {
                            onMouseOver: (e) => handleMouseOver(e, node), onMouseOut: handleMouseOut,
                            onClick: () => handleClick(node), style: { cursor: 'pointer' }, fill: node.color || '#999'
                        };

                        return (
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
                {/* Render Links on Top */}
                <g>
                    {links.map((link, i) => {
                        const sourceNode = nodeMap.get(link.source);
                        const targetNode = nodeMap.get(link.target);
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
                <defs>
                    <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#999" />
                    </marker>
                </defs>
            </svg>

            {/* Tooltip Element */}
            {tooltip && (
                <div
                    style={{
                        position: 'fixed',
                        top: tooltip.y + 15,
                        left: tooltip.x + 15,
                        background: 'black',
                        color: 'white',
                        padding: '8px',
                        borderRadius: '4px',
                        pointerEvents: 'none',
                        fontSize: '12px',
                        zIndex: 100
                    }}
                >
                    {tooltip.content && Object.entries(tooltip.content).map(([key, value]) => (
                        <div key={key}><strong>{key}:</strong> {value}</div>
                    ))}
                </div>
            )}
        </Card>
    );
}
