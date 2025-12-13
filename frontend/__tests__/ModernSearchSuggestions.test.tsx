import { render, screen } from '@testing-library/react'
import ModernSearchSuggestions from '@/components/ModernSearchSuggestions'

describe('ModernSearchSuggestions', () => {
  const mockSuggestions = [
    'Eiffel Tower',
    'Louvre Museum',
    'Arc de Triomphe',
  ]

  it('renders all suggestions', () => {
    const mockOnSelect = jest.fn()
    render(
      <ModernSearchSuggestions 
        suggestions={mockSuggestions} 
        onSelect={mockOnSelect} 
      />
    )

    mockSuggestions.forEach(suggestion => {
      expect(screen.getByText(suggestion)).toBeInTheDocument()
    })
  })

  it('calls onSelect when a suggestion is clicked', () => {
    const mockOnSelect = jest.fn()
    render(
      <ModernSearchSuggestions 
        suggestions={mockSuggestions} 
        onSelect={mockOnSelect} 
      />
    )

    const firstSuggestion = screen.getByText('Eiffel Tower')
    firstSuggestion.click()

    expect(mockOnSelect).toHaveBeenCalledWith('Eiffel Tower')
    expect(mockOnSelect).toHaveBeenCalledTimes(1)
  })

  it('renders nothing when no suggestions provided', () => {
    const mockOnSelect = jest.fn()
    const { container } = render(
      <ModernSearchSuggestions 
        suggestions={[]} 
        onSelect={mockOnSelect} 
      />
    )

    // Component returns null when empty, so firstChild is null
    expect(container.firstChild).toBeNull()
  })
})
